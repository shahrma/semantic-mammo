import copy
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torchvision.datasets as datasets
import wandb
import argparse
import pickle

import torch
import numpy as np
import random

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

def explainability(model, val_loader, criteria, device, workspace_path, tconfig, categories):
    use_meta = tconfig.get('use_meta', True)

    routing_config = tconfig.get('routing_criterion', None)

    checkpoint_path = os.path.join(workspace_path, 'best_model.pth')
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    print(f'Loaded checkpoint : {checkpoint_path}')

    # Find multiple conv layers for GradCAM comparison
    # Handle DataParallel wrapping
    base_model = model.module if hasattr(model, 'module') else model

    # Collect conv layers at different stages for comparison
    conv_layers_by_stage = {}
    all_modules = dict(base_model.named_modules())

    for name, module in base_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Extract stage info from layer name (e.g., "model.features.7.2.block.0")
            parts = name.split('.')
            # Find the features.X part to identify stage
            for i, part in enumerate(parts):
                if part == 'features' and i + 1 < len(parts):
                    try:
                        stage_num = int(parts[i + 1])
                        stage_key = f"Stage{stage_num}"
                        # Keep track of layers per stage (will keep last one)
                        conv_layers_by_stage[stage_key] = (name, module)
                    except ValueError:
                        pass
                    break

    # Select representative layers for GradCAM comparison
    # Typically: early stage, middle stage, and last stage
    target_layers = {}
    stage_keys = sorted(conv_layers_by_stage.keys(), key=lambda x: int(x.replace('Stage', '')))

    if len(stage_keys) >= 1:
        # Last stage (most semantic)
        last_stage = stage_keys[-1]
        target_layers[last_stage] = conv_layers_by_stage[last_stage][1]

    if len(stage_keys) >= 3:
        # Middle stage
        mid_idx = len(stage_keys) // 2
        mid_stage = stage_keys[mid_idx]
        target_layers[mid_stage] = conv_layers_by_stage[mid_stage][1]

    if len(stage_keys) >= 5:
        # Earlier stage (higher resolution)
        early_idx = len(stage_keys) // 4
        early_stage = stage_keys[early_idx]
        target_layers[early_stage] = conv_layers_by_stage[early_stage][1]

    print("Selected GradCAM layers:")
    for layer_name, layer_module in target_layers.items():
        print(f"  {layer_name}: {layer_module}")

    if target_layers:
        from core.inference_analysis import gradcam_on_full_dataset

        # Get the dataset object for mask loading
        dataset = val_loader.dataset

        # Generate GradCAM for all samples in val_loader
        limited_output = tconfig.get('gradcam_limited_output', False)
        gradcam_on_full_dataset(
            model=base_model,
            val_loader=val_loader,
            device=device,
            target_layers=target_layers,
            out_dir=os.path.join(workspace_path, "gradcampp"),
            categories=categories,
            dataset=dataset,
            use_meta=use_meta,
            positive_class_index=0,
            limited_output=limited_output,
        )
    else:
        print("Warning: Could not find Conv2d layers for GradCAM")

    # Run meta impact analysis if using metadata
    if use_meta:
        print("\n" + "=" * 60)
        print("Running Meta Impact Analysis...")
        print("=" * 60)
        from core.meta_impact_analysis import MetaImpactAnalyzer

        analyzer = MetaImpactAnalyzer(base_model, device, categories)
        meta_analysis_results = analyzer.run_all_analyses(
            val_loader,
            output_dir=os.path.join(workspace_path, "meta_impact_analysis")
        )




def meta_analysis(model, val_loader, criteria, device, workspace_path, tconfig, categories):
    use_meta = tconfig.get('use_meta', True)

    routing_config = tconfig.get('routing_criterion', None)

    checkpoint_path = os.path.join(workspace_path, 'best_model.pth')
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    print(f'Loaded checkpoint : {checkpoint_path}')

    # Find multiple conv layers for GradCAM comparison
    # Handle DataParallel wrapping
    base_model = model.module if hasattr(model, 'module') else model

    from core.meta_impact_analysis import MetaImpactAnalyzer
    analyzer = MetaImpactAnalyzer(base_model, device, categories)

    # ablation_df, ablation_summary = analyzer.meta_monotonicity_test(
    #     val_loader, output_dir=os.path.join(workspace_path, "meta_monotonicity_test")
    # )

    ablation_df, ablation_summary = analyzer.meta_boundary_sensitivity_test(
        val_loader, output_dir=os.path.join(workspace_path, "meta_boundary_sensitivity_test")
    )

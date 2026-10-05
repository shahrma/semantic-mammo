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

import pandas as pd
from core.LDN_wrapper import depth_replacer, LDN_warpping

import re

def print_trainable(model):
    for name, params in model.named_parameters():
        print(f"{name:60s}  {params.requires_grad}")

def set_trainable(model, patterns, trainable=True):
    """
    Freezes (requires_grad=False) all parameters in modules whose name
    matches any regex pattern in patterns.

    Args:
        model: torch.nn.Module
        patterns: list of regex strings
    """

    params_to_change = []
    for name, params in model.named_parameters():
        params_to_change.append(name)

    import fnmatch
    if patterns is not None:
        matched_params = []
        for pattern in patterns:
            matched_params.extend(fnmatch.filter(params_to_change, pattern))
        params_to_change = sorted(list(set(matched_params)))

    for name, params in model.named_parameters():
        if name in params_to_change:
            print(f"{'Un-Freezing' if trainable else 'Freezing'}: {name}")
            params.requires_grad = trainable

    return model


import torch
import torch.nn as nn
import torchvision.models as models
import importlib
from functools import reduce

def get_torchvision_model_dictionaries():
    classifier_paths = {
        # EfficientNet-V2
        "efficientnet_v2_s": ["classifier", 1],
        "efficientnet_v2_m": ["classifier", 1],
        "efficientnet_v2_l": ["classifier", 1],

        # EfficientNet
        "efficientnet_b0": ["classifier", 1],
        "efficientnet_b1": ["classifier", 1],
        "efficientnet_b2": ["classifier", 1],
        "efficientnet_b3": ["classifier", 1],
        "efficientnet_b4": ["classifier", 1],
        "efficientnet_b5": ["classifier", 1],
        "efficientnet_b6": ["classifier", 1],
        "efficientnet_b7": ["classifier", 1],

        # MobileNet
        "mobilenet_v2": ["classifier", 1],
        "mobilenet_v3_small": ["classifier", 3],
        "mobilenet_v3_large": ["classifier", 3],

        # ResNet
        "resnet18": ["fc"],
        "resnet34": ["fc"],
        "resnet50": ["fc"],
        "resnet101": ["fc"],
        "resnet152": ["fc"],
        "resnext50_32x4d": ["fc"],
        "resnext101_32x8d": ["fc"],
        "wide_resnet50_2": ["fc"],
        "wide_resnet101_2": ["fc"],

        # DenseNet
        "densenet121": ["classifier"],
        "densenet161": ["classifier"],
        "densenet169": ["classifier"],
        "densenet201": ["classifier"],

        # ViT
        "vit_b_16": ["heads", "head"],
        "vit_b_32": ["heads", "head"],
        "vit_l_16": ["heads", "head"],
        "vit_l_32": ["heads", "head"],
        "vit_h_14": ["heads", "head"],

        # ConvNeXt
        "convnext_tiny": ["classifier", 2],
        "convnext_small": ["classifier", 2],
        "convnext_base": ["classifier", 2],
        "convnext_large": ["classifier", 2],

        # RegNet
        "regnet_y_400mf": ["fc"],
        "regnet_y_800mf": ["fc"],
        "regnet_y_1_6gf": ["fc"],
        "regnet_y_3_2gf": ["fc"],
        "regnet_y_8gf": ["fc"],
        "regnet_y_16gf": ["fc"],
        "regnet_y_32gf": ["fc"],
        "regnet_y_128gf": ["fc"],
        "regnet_x_400mf": ["fc"],
        "regnet_x_800mf": ["fc"],
        "regnet_x_1_6gf": ["fc"],
        "regnet_x_3_2gf": ["fc"],
        "regnet_x_8gf": ["fc"],
        "regnet_x_16gf": ["fc"],
        "regnet_x_32gf": ["fc"],

        # SqueezeNet
        "squeezenet1_0": ["classifier", 1],
        "squeezenet1_1": ["classifier", 1],
    }

    weight_enum_map = {
        # EfficientNet-V2
        "efficientnet_v2_s": "EfficientNet_V2_S_Weights",
        "efficientnet_v2_m": "EfficientNet_V2_M_Weights",
        "efficientnet_v2_l": "EfficientNet_V2_L_Weights",

        # EfficientNet
        "efficientnet_b0": "EfficientNet_B0_Weights",
        "efficientnet_b1": "EfficientNet_B1_Weights",
        "efficientnet_b2": "EfficientNet_B2_Weights",
        "efficientnet_b3": "EfficientNet_B3_Weights",
        "efficientnet_b4": "EfficientNet_B4_Weights",
        "efficientnet_b5": "EfficientNet_B5_Weights",
        "efficientnet_b6": "EfficientNet_B6_Weights",
        "efficientnet_b7": "EfficientNet_B7_Weights",

        # MobileNet
        "mobilenet_v2": "MobileNet_V2_Weights",
        "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
        "mobilenet_v3_large": "MobileNet_V3_Large_Weights",

        # ResNet
        "resnet18": "ResNet18_Weights",
        "resnet34": "ResNet34_Weights",
        "resnet50": "ResNet50_Weights",
        "resnet101": "ResNet101_Weights",
        "resnet152": "ResNet152_Weights",
        "resnext50_32x4d": "ResNeXt50_32X4D_Weights",
        "resnext101_32x8d": "ResNeXt101_32X8D_Weights",
        "wide_resnet50_2": "Wide_ResNet50_2_Weights",
        "wide_resnet101_2": "Wide_ResNet101_2_Weights",

        # DenseNet
        "densenet121": "DenseNet121_Weights",
        "densenet161": "DenseNet161_Weights",
        "densenet169": "DenseNet169_Weights",
        "densenet201": "DenseNet201_Weights",

        # ViT
        "vit_b_16": "ViT_B_16_Weights",
        "vit_b_32": "ViT_B_32_Weights",
        "vit_l_16": "ViT_L_16_Weights",
        "vit_l_32": "ViT_L_32_Weights",
        "vit_h_14": "ViT_H_14_Weights",

        # ConvNeXt
        "convnext_tiny": "ConvNeXt_Tiny_Weights",
        "convnext_small": "ConvNeXt_Small_Weights",
        "convnext_base": "ConvNeXt_Base_Weights",
        "convnext_large": "ConvNeXt_Large_Weights",

        # RegNet
        "regnet_y_400mf": "RegNet_Y_400MF_Weights",
        "regnet_y_800mf": "RegNet_Y_800MF_Weights",
        "regnet_y_1_6gf": "RegNet_Y_1_6GF_Weights",
        "regnet_y_3_2gf": "RegNet_Y_3_2GF_Weights",
        "regnet_y_8gf": "RegNet_Y_8GF_Weights",
        "regnet_y_16gf": "RegNet_Y_16GF_Weights",
        "regnet_y_32gf": "RegNet_Y_32GF_Weights",
        "regnet_y_128gf": "RegNet_Y_128GF_Weights",
        "regnet_x_400mf": "RegNet_X_400MF_Weights",
        "regnet_x_800mf": "RegNet_X_800MF_Weights",
        "regnet_x_1_6gf": "RegNet_X_1_6GF_Weights",
        "regnet_x_3_2gf": "RegNet_X_3_2GF_Weights",
        "regnet_x_8gf": "RegNet_X_8GF_Weights",
        "regnet_x_16gf": "RegNet_X_16GF_Weights",
        "regnet_x_32gf": "RegNet_X_32GF_Weights",

        # SqueezeNet
        "squeezenet1_0": "SqueezeNet1_0_Weights",
        "squeezenet1_1": "SqueezeNet1_1_Weights",
    }

    return classifier_paths, weight_enum_map

def get_torchvision_model(model_name: str, weight_name: str, num_classes: int, use_pretrained: bool = True, model_args: dict = None):
    model_args = model_args or {}

    classifier_paths, weight_enum_map = get_torchvision_model_dictionaries()

    if model_name not in classifier_paths or model_name not in weight_enum_map:
        raise ValueError(f"Model '{model_name}' is not supported.")

    # Dynamically import and get weights
    weight_enum_name = weight_enum_map[model_name]
    try:
        weights_module = importlib.import_module("torchvision.models")
        weight_class = getattr(weights_module, weight_enum_name)
        weights = getattr(weight_class, weight_name) if use_pretrained else None
    except AttributeError:
        raise ValueError(f"Could not find weights '{weight_name}' for model '{model_name}'")

    # Add weights to model args
    model_args["weights"] = weights

    # Load the model constructor and create the model
    try:
        model_fn = getattr(models, model_name)
        model = model_fn(**model_args)
    except AttributeError:
        raise ValueError(f"Model constructor '{model_name}' not found in torchvision.models")

    # Access the classifier layer and replace it
    classifier_path = classifier_paths[model_name]
    classifier_parent = reduce(getattr, [model] + classifier_path[:-1])
    classifier_attr = classifier_path[-1]

    if isinstance(classifier_attr, int):
        in_features = classifier_parent[classifier_attr].in_features
        classifier_parent[classifier_attr] = nn.Linear(in_features, num_classes)
    else:
        old_layer = getattr(classifier_parent, classifier_attr)
        in_features = old_layer.in_features
        setattr(classifier_parent, classifier_attr, nn.Linear(in_features, num_classes))

    return model

def get_model(mconfig, in_channels, num_classes, use_pretrained, categories=None):
    # Define model
    if mconfig['model_type'] == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None)
        in_channels = model.conv1.in_channels if in_channels is None else in_channels
        model.conv1 = nn.Conv2d(in_channels, model.conv1.out_channels,
                                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif mconfig['model_type'] == 'resnet50-orig':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
        in_channels = model.conv1.in_channels if in_channels is None else in_channels
        model.conv1 = nn.Conv2d(in_channels, model.conv1.out_channels,
                                kernel_size=model.conv1.kernel_size, stride=model.conv1.stride,
                                padding=model.conv1.padding, bias=model.conv1.bias)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif mconfig['model_type'] == 'vit_b_16':
        dropout = mconfig.get('dropout', 0.0)
        attention_dropout = mconfig.get('attention_dropout', 0.0)

        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if use_pretrained else None,
                                dropout=dropout,attention_dropout=attention_dropout)

        in_features = model.heads.head.in_features
        model.heads = nn.Sequential(nn.Linear(in_features, num_classes))

    elif mconfig['model_type'] == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
        in_channels = model.conv1.in_channels if in_channels is None else in_channels
        model.conv1 = nn.Conv2d(in_channels, model.conv1.out_channels,
                                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif mconfig['model_type'] == 'wrapped_convnext_base':
        from core.wrapped_models.wrapped_convnext import WrappedConvNeXt
        replacing_list = mconfig.get('replacing_list',dict())
        model = WrappedConvNeXt(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if use_pretrained else None,
                                replacing_list=replacing_list,
                                num_classes=num_classes,
                                categories=categories)
    elif mconfig['model_type'] == 'wrapped_convnext_se_gate':
        from core.se_gate.wrapped_convnext_se_gate import WrappedConvNeXtSEGate
        descriptor_groups = mconfig.get('descriptor_groups', None)
        se_reduction_ratio = mconfig.get('se_reduction_ratio', 16)
        model = WrappedConvNeXtSEGate(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if use_pretrained else None,
                                      num_classes=num_classes,
                                      categories=categories,
                                      descriptor_groups=descriptor_groups,
                                      se_reduction_ratio=se_reduction_ratio)

    elif mconfig['model_type'] == 'multitask_convnext':
        from core.multitask.multitask_convnext import MultiTaskConvNeXt
        descriptor_groups = mconfig.get('descriptor_groups', None)
        model = MultiTaskConvNeXt(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if use_pretrained else None,
                                  num_classes=num_classes,
                                  categories=categories,
                                  descriptor_groups=descriptor_groups)

    elif mconfig['model_type'] == 'wrapped_convnext_concat':
        from core.late_fusion.wrapped_convnext_concat import WrappedConvNeXtConcat
        descriptor_groups = mconfig.get('descriptor_groups', None)
        model = WrappedConvNeXtConcat(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if use_pretrained else None,
                                      num_classes=num_classes,
                                      categories=categories,
                                      descriptor_groups=descriptor_groups)

    elif mconfig['model_type'] == 'extended_convnext':
        from core.wrapped_models.extended_convnext import ExtendedConvNeXt
        replacing_list = mconfig.get('replacing_list',dict())
        model = ExtendedConvNeXt(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if use_pretrained else None,
                                replacing_list=replacing_list,
                                num_classes=num_classes)

    elif mconfig['model_type'] == 'cross_attention':
        from core.cross_attention.cross_attention_convnext import CrossAttentionFusionModel
        descriptor_groups = mconfig.get('descriptor_groups', None)
        num_heads = mconfig.get('num_heads', 8)
        model = CrossAttentionFusionModel(
            weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if use_pretrained else None,
            num_classes=num_classes,
            categories=categories,
            descriptor_groups=descriptor_groups,
            num_heads=num_heads,
        )

    else:
        raise f"model_type {mconfig['model_type']} was not implemented in code"
    # Modify the last layer of the model

    return model

def build_classification_model(mconfig, num_classes, categories):
    use_pretrained = mconfig.get('use_pretrained', True)
    reduction_layers = mconfig.get('reduction_layers', [])
    warped_layers = mconfig.get('warped_layers', [])
    noise_type = mconfig.get('noise_type', 'gaussian')
    in_channels = mconfig.get('in_channels', None)
    freeze_patterns = mconfig.get('freeze_patterns', None)
    trainable_patterns = mconfig.get('trainable_patterns', None)

    model_source = mconfig.get('model_source', None)

    if model_source == 'torchvision':
        model = get_torchvision_model(mconfig['model_type'],mconfig['model_weights'],
                                      num_classes=num_classes, use_pretrained=use_pretrained,
                                      model_args=mconfig['model_args'])
    else:
        model = get_model(mconfig, in_channels, num_classes, use_pretrained, categories)

    reduction_factor = mconfig.get('reduction_factor', None)
    if reduction_factor is not None:
        print(f'Depth reduction by factor {reduction_factor}')
        model = depth_replacer(model, layers=reduction_layers, factor=reduction_factor)

    model_wrapper = mconfig.get('model_wrapper', None)
    if model_wrapper == 'LDN':
        print(f'Wrapping model with {model_wrapper}')
        model = LDN_warpping(model, layers=warped_layers, noise_type=noise_type)

    if freeze_patterns is not None:
        model = set_trainable(model, freeze_patterns, trainable=False)

    if trainable_patterns is not None:
        model = set_trainable(model, trainable_patterns, trainable=True)

    # print_trainable(model)

    return model

if __name__ == '__main__':
    import torchvision.models as models
    from torchinfo import summary
    from copy import deepcopy

    model = get_torchvision_model(
        model_name="vit_b_16",
        weight_name="IMAGENET1K_V1",
        num_classes=10,
        use_pretrained=True
    )

    models_list = ['efficientnet_b2','mobilenet_v2','mobilenet_v3_large','resnet101','vit_b_32','convnext_base','densenet121','vit_l_32']

    for model_name in models_list:
        model = get_torchvision_model(
            model_name=model_name,
            weight_name="IMAGENET1K_V1",
            num_classes=10,
            use_pretrained=True
        )
        print(model)



    print(model)


    model = get_torchvision_model('mobilenet_v3_large','IMAGENET1K_V1', num_classes=5, use_pretrained = True)
    print(model)

    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    print(model)

    model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    print(model)

    model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    print(model)

    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    print(model)
    print_trainable(model)

    frozen_patterns = [f'*']
    set_trainable(model, frozen_patterns, trainable=False)
    print_trainable(model)
    trainable_patterns = ['encoder.layers.encoder_layer_*.ln*', 'encoder.ln','heads.head']
    set_trainable(model, trainable_patterns, trainable=True)
    print_trainable(model)
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
import copy

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

import pandas as pd

from core.configs import load_yaml_config, set_nested_value_from_str_path

# from core.datasets import get_dataset_CBIS_DDSM
from core.datasets import get_ROI_dataset, get_categories
from core.models import build_classification_model
from core.engine import train, evaluation,  meta_evaluation
from core.analysis import explainability, meta_analysis

def load_categories(dconfig, add_unknown):
    # os.makedirs(workspace_path, exist_ok=True)
    filename = os.path.join('./configs/categories', 'categories01.json')
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            categories = json.load(f)
    else:
        with open(filename, "w", encoding="utf-8") as f:
            categories = get_categories(dconfig, add_unknown=add_unknown)
            json.dump(categories, f, ensure_ascii=False, indent=4)

    return categories

def main(workspace_path, phase, config):
    os.makedirs(workspace_path, exist_ok=True)

    used_dataset = config.get('used_dataset', None)
    dconfig = {k:v for k,v in config['datasets'].items() if k in used_dataset}
    pconfig = config['preprocess']
    mconfig = config['model']
    tconfig = config['training']

    add_unknown = pconfig.get('add_unknown', False)
    full_categories = load_categories(dconfig, add_unknown)
    target_col = dconfig.get('target_col', 'pathology')
    model_categories = [target_col]
    if 'descriptor_groups' in mconfig:
        model_categories += list(mconfig['descriptor_groups'])
    elif 'replacing_list' in mconfig:
        model_categories += list(mconfig['replacing_list'].keys())

    categories = {k:v for k,v in full_categories.items() if k in model_categories}

    train_dataset, val_dataset, num_classes = get_ROI_dataset(dconfig, pconfig, categories)

    # Create data loaders for the train and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=tconfig['batch_size'], shuffle=True, num_workers=tconfig.get('num_workers', 8))
    val_loader = DataLoader(val_dataset, batch_size=tconfig['batch_size'], shuffle=False, num_workers=tconfig.get('num_workers', 8))

    model = build_classification_model(mconfig, num_classes, categories)
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model = torch.nn.DataParallel(model)

    # Unfreeze all the layers and fine-tune the entire network for a few more epochs
    for param in model.parameters():
        param.requires_grad = True

    # Define the loss function and optimizer
    criteria = tconfig.get('criteria', {'main':'CrossEntropyLoss'})

    if 'train' in phase:
        optimizer = None
        optimizer_type = tconfig.get('optimizer_type', None)
        if optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=tconfig['learning_rate'], weight_decay=tconfig['weight_decay'])
        elif optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=tconfig['learning_rate'], momentum=tconfig['momentum'], weight_decay=tconfig['weight_decay'])

        train(model, train_loader, val_loader, criteria, optimizer,
              device=device, workspace_path=workspace_path, tconfig=tconfig,categories=categories)

    if 'evaluation' in phase:
        evaluation(model, val_loader, criteria, device, workspace_path, tconfig,categories=categories)

    if 'explainability' in phase:
        explainability(model, val_loader, criteria, device, workspace_path, tconfig,categories=categories)

    if 'meta_analysis' in phase:
        meta_analysis(model, val_loader, criteria, device, workspace_path, tconfig,categories=categories)


    if 'meta_evaluation' in phase:
        run_meta_evaluation(config, dconfig, pconfig, mconfig, tconfig, full_categories, criteria, device, workspace_path)


def run_meta_evaluation(config, dconfig, pconfig, mconfig, tconfig, full_categories, criteria, device, workspace_path):
    """
    Run meta evaluation: train tabular models on metadata features.
    Uses the same metadata descriptors specified in the episode config.
    """
    # Get the descriptors used in the model
    if 'descriptor_groups' in mconfig:
        meta_descriptors = list(mconfig['descriptor_groups'])
    elif 'replacing_list' in mconfig:
        meta_descriptors = list(mconfig['replacing_list'].keys())
    else:
        meta_descriptors = []

    # Filter categories to only include the descriptors used in the episode + pathology
    meta_categories = {k: v for k, v in full_categories.items() if k in meta_descriptors or k == 'pathology'}

    # Get dataset name for output files
    used_dataset = config.get('used_dataset', [])
    dataset_name = config.get('args', {}).get('dataset', '_'.join(used_dataset))

    # Create dataloaders with the filtered categories for meta_evaluation
    meta_train_dataset, meta_val_dataset, _ = get_ROI_dataset(dconfig, pconfig, meta_categories)
    meta_train_loader = DataLoader(meta_train_dataset, batch_size=tconfig['batch_size'], shuffle=False, num_workers=tconfig.get('num_workers', 8))
    meta_val_loader = DataLoader(meta_val_dataset, batch_size=tconfig['batch_size'], shuffle=False, num_workers=tconfig.get('num_workers', 8))

    meta_evaluation(meta_train_loader, meta_val_loader, criteria, device, workspace_path, tconfig, categories=meta_categories, dataset_name=dataset_name)


import subprocess
def get_git_tag():
    return subprocess.check_output(["git", "describe", "--tags", "--always"]).strip().decode()

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_mode', type=str, choices=['online', 'offline', 'disabled', 'dryrun'], default='online',
                        help="Set mode for Weights and Biases (online, offline, disabled or dryrun)")
    parser.add_argument('--project', type=str, default='mammo-roi-classifier')
    parser.add_argument('--workspace', type=str, default='../../workspace/mammo-roi-classifier')
    parser.add_argument('--dataset', type=str, default='ddsm_krois_001')
    parser.add_argument('--episode', type=str)
    parser.add_argument('--group', type=str, default=None)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--phase', type=str, nargs='+', default=['train', 'evaluation'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--git_tag', type=str, default=None, help="DO NOT CHANGE")
    parser.add_argument('--params', type=str, nargs='+', default=None)

    args = parser.parse_args()

    args.git_tag = get_git_tag() if args.git_tag is None else args.git_tag

    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU training
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures reproducibility


def prepare_configurations(args):
    config = dict()
    config['args'] = copy.deepcopy(vars(args))
    dataset_configs = load_yaml_config(f'configs/datasets/{args.dataset}.yaml')
    episode_configs = load_yaml_config(f'configs/episodes/{args.episode}.yaml')
    config.update(dataset_configs)
    config.update(episode_configs)

    if args.params is not None:
        for p in args.params:
            keys, value = p.split(':')
            set_nested_value_from_str_path(config, keys, value)

    if args.seed is not None:
        set_seed(args.seed)

    return config

if __name__ == '__main__':
    args = init_args()
    config = prepare_configurations(args)

    print(json.dumps(config, indent=4))
    model_type = config['model']['model_type']

    name_list = [args.dataset, args.episode, model_type, args.name,  args.seed,  args.git_tag]
    name_list = [k for k in name_list if k is not None and k != '']
    experiment_name = '-'.join([f'{s}' for s in name_list])

    group_name = '-'.join([f'{s}' for s in name_list[0:4]]) if args.group is None else args.group

    workspace_path = os.path.join(args.workspace, group_name, experiment_name)
    os.makedirs(workspace_path, exist_ok=True)
    with open(os.path.join(workspace_path,"configuration.json"), "w") as f:
        json.dump(config, f, indent=4)

    run = wandb.init(project=f'{args.project}', group=f'{group_name}', name=f'{experiment_name}',
               config=config, mode=args.wandb_mode, dir=workspace_path)

    main(workspace_path, args.phase, config)

    run.finish()

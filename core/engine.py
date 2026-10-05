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

import pandas as pd

from core.LDN_wrapper import collect_model_stats
from core.LDN_utils import produce_stats, get_routing_criterion, WarmUpLR


# loss = criterion(outputs, labels)
# if routing_criterion is not None:
#     uniform_loss_value, probs = routing_criterion(epoch)
#     if uniform_loss_value is not None:
#         loss += uniform_loss_value

def calculate_multiple_loss(outputs, criteria, targets):
    loss = 0.0
    main_preds=None
    verbose = []

    for output in outputs:
        group = output['group']
        predicted =  output['predicted']
        name = output['name']

        if str(group) in criteria:
            criteria_type = criteria[group].get('type',criteria[group]) # For backward compatible with EXP0030 - can be removed later
            criteria_weight = criteria[group].get('weight',1.0)
            criteria_loss = None
            if criteria_type == 'CrossEntropyLoss':
                labels = targets[group]
                criterion = nn.CrossEntropyLoss()
                criteria_loss = criterion(predicted, labels)

                _, preds_ = torch.max(predicted, 1)
            elif criteria_type == 'BCEWithLogitsLoss':
                labels = targets[group]
                criterion = nn.BCEWithLogitsLoss()
                criteria_loss = criterion(predicted, labels)

                probs = torch.sigmoid(predicted)
                preds_ = (probs > 0.5).int()
            else:
                raise f'{criteria_type} is no valid criterion'

            loss += criteria_weight * criteria_loss

            preds_ = preds_.detach().cpu().numpy()
            predicted_ = predicted.detach().cpu().numpy()
            labels_= labels.detach().cpu().numpy()
            criteria_loss_ = criteria_loss.item()
            verbose_ = {'group':group, 'name':name ,'predicted':predicted_, 'preds':preds_, 'labels':labels_,'criteria_loss':criteria_loss_,'criteria_weight':criteria_weight}
            verbose.append(verbose_)

        if group == 'main':
            main_preds = copy.deepcopy(preds_)
            main_predicted =  copy.deepcopy(predicted_)
    # main_preds = [item for item in verbose if item['group'] == 'main'][0]['preds']

    return loss, main_preds , main_predicted, verbose


def train_epoch(epoch, train_loader,
                optimizer, warmup_scheduler, warmup_epochs,
                model, criteria,  device,use_meta=True):
    model.train()

    running_loss = 0.0
    running_corrects = 0
    verbose_loss = {}
    all_probs = []
    # Iterate over the batches of the train loader
    for inputs, labels, meta, index in train_loader:
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)
        meta = {k: v.to(device) for k, v in meta.items()}

        optimizer.zero_grad()

        outputs = model(inputs, meta) if use_meta else model(inputs)
        outputs = outputs if isinstance(outputs, list) else [{'group':'main', 'name': 'main', 'predicted': outputs}]

        targets = {'main': labels}
        targets.update(meta)

        loss, main_preds, main_predicted ,verbose_loss = calculate_multiple_loss(outputs, criteria, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(main_preds== labels.data.cpu()).item()

        if epoch <= warmup_epochs:
            warmup_scheduler.step()

    # Calculate the train loss and accuracy
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = running_corrects / len(train_loader.dataset)

    return train_loss, train_acc, verbose_loss


def evaluate_epoch(epoch, val_loader, model,criteria, device, use_meta=True):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    all_probs = []
    all_scalars = []

    predictions = []
    labels_list = []
    # Iterate over the batches of the validation loader
    with torch.no_grad():
        for inputs, labels, meta, index in val_loader:

            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)
            meta = {k: v.to(device) for k, v in meta.items()}

            # Forward pass
            outputs = model(inputs, meta) if use_meta else model(inputs)
            outputs = outputs if isinstance(outputs, list) else [{'group':'main', 'name': 'main', 'predicted': outputs}]

            targets = {'main': labels}
            targets.update(meta)
            try:
                loss, main_preds, main_predicted, verbose = calculate_multiple_loss(outputs, criteria, targets)
            except:
                print('evaluate')
                print(outputs, criteria, targets)

            probs, scalars = collect_model_stats(model)
            if probs is not None:
                all_probs.append(probs.detach().cpu())

            if scalars is not None:
                all_scalars.append(scalars.detach().cpu())

            # Update the running loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(main_preds == labels.data.cpu()).item()

            # main_preds = [item for item in verbose if item['group'] == 'main'][0]['predicted']
            predictions.append(main_predicted)  # Save predictions
            labels_list.append(labels.cpu().numpy())

    # Calculate the validation loss and accuracy
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = running_corrects / len(val_loader.dataset)

    labels_list = np.hstack(labels_list)
    predictions = softmax(np.vstack([np.stack(p) for p in predictions]))
    predicted = np.argmax(predictions, axis=1)
    verbose = {'targets':labels_list, 'predictions': predictions, 'predicted':predicted}
    if len(all_probs) > 0:
        probs = produce_stats(all_probs, 'probs')
        verbose.update({'probs' :probs })
    if len(all_scalars) > 0:
        scalars = pd.DataFrame(all_scalars[0].numpy(), columns=[f'scalar_{i}' for i in range(scalars.shape[1])])
        verbose.update({'scalars': scalars })

    return val_loss, val_acc, verbose


def softmax(x):
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def define_scheduler(scheduler_type,params, optimizer):
    scheduler = None
    if scheduler_type == 'MultiStepLR':
        milestones = params.get('milestones', None)
        gamma = params.get('gamma', 0.2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif scheduler_type == 'ReduceLROnPlateau':
        factor = params.get('factor', 0.5)
        patience = params.get('patience', 10)
        mode = params.get('mode', 10)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=True)
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        T_0 = params.get('T_0', 10)
        T_mult = params.get('T_mult', 2)
        eta_min = params.get('eta_min', 0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=-1)
    else:
        raise f'{scheduler_type} is not valid scheduler'

    return scheduler


from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score
def calc_statistics(targets, predictions, categories, category_name = None):
    y_pred = np.argmax(predictions, axis=1)

    stats = {}
    stats['f1'] = f1_score(targets, y_pred)
    stats['acc'] = accuracy_score(targets, y_pred)
    stats['recall'] = recall_score(targets, y_pred)

    category_name = list(categories.keys())[0] if category_name is None else category_name
    for name, value in categories[category_name].items():
        try:
            stats[f'auc:{category_name}:{name}'] = roc_auc_score(targets, predictions[:,value])
        except:
            print(f'auc:{category_name}:{name} does not have value {value} ' )

    return stats

def train(model, train_loader, val_loader, criteria, optimizer, device, workspace_path, tconfig,categories):
    warmup_epochs = tconfig.get('warmup_epochs', 1)
    num_epochs = tconfig.get('num_epochs', None)
    scheduler_type = tconfig.get('scheduler_type', None)
    routing_config = tconfig.get('routing_criterion', None)
    use_meta = tconfig.get('use_meta', True)

    os.makedirs(workspace_path, exist_ok=True)
    os.makedirs(os.path.join(workspace_path, 'info'), exist_ok=True)
    with open(os.path.join(workspace_path, f'categories.json'), "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=4)

    iter_per_epoch = len(train_loader)

    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warmup_epochs)
    train_scheduler =define_scheduler(scheduler_type, tconfig[scheduler_type], optimizer)

    # routing_criterion = get_routing_criterion(model, routing_config)

    best_val_acc = 0.0
    # Train the model for the specified number of epochs

    artifact = wandb.Artifact(name="model", type="model", description=" ")
    best_info_file  = None
    for epoch in range(num_epochs):
        train_loss, train_acc, train_verbose = train_epoch(epoch, train_loader,
                                            optimizer, warmup_scheduler, warmup_epochs,
                                            model, criteria,  device,use_meta=use_meta)

        param_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]

        if epoch > warmup_epochs:
            train_scheduler.step(epoch)

        val_loss, val_acc, val_verbose = evaluate_epoch(epoch, val_loader, model, criteria, device,use_meta=use_meta)
        stats = calc_statistics(targets = val_verbose['targets'], predictions = val_verbose['predictions'], categories=categories, category_name='pathology')
        stats = {f'val_{k}':v for k,v in stats.items()}

        model_info = {
            "epoch": epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'param_lr': param_lr,
            "predictions": val_verbose['predictions'],
            "targets": val_verbose['targets']
        }

        info_file = os.path.join(workspace_path, 'info', f"info_{epoch + 1:04d}.pkl")
        with open(info_file, "wb") as f:
            pickle.dump(model_info, f)
        artifact.add_file(info_file)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(workspace_path, f'best_model.pth'))
            print(f"New best model saved with val_acc: {best_val_acc:.4f}")

            best_info_file = os.path.join(workspace_path, 'info', f"best_model_info.pkl")
            with open(best_info_file, "wb") as f:
                pickle.dump(model_info, f)

            with open(os.path.join(workspace_path, f"best_model_optimizer.pkl"), "wb") as f:
                model_info.update({"optimizer_state_dict": optimizer.state_dict()})
                pickle.dump(model_info, f)


        log_dict = {'epoch': epoch + 1,
                   'lr': param_lr,
                   'train_loss': train_loss,
                   'train_acc': train_acc,
                   'val_loss': val_loss,
                   'val_acc': val_acc}

        log_dict.update(stats)
        if 'probs' in val_verbose.keys():
            log_dict.update({"probs": wandb.Table(dataframe=val_verbose['probs'])})
            probs_dict = [{f"{data['probs']}_{k}": v for k, v in data.items() if k != 'probs'} for data in val_verbose['probs'].to_dict('records')]
            merged = {k: v for d in probs_dict for k, v in d.items()}
            log_dict.update(merged)
        if 'scalars' in val_verbose.keys():
            log_dict.update({"scalars": wandb.Table(dataframe=val_verbose['scalars'])})
            scalar_dict = [{f"{k}-{idx}": v for  (k, v) in data.items() if k != 'scalars'} for idx, data in enumerate(val_verbose['scalars'].to_dict('records'))]
            merged = {k: v for d in scalar_dict for k, v in d.items()}
            log_dict.update(merged)

        wandb.log(log_dict)

        # Print the epoch results
        print(f'Epoch [{epoch + 1}/{num_epochs}], train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, '
              f'val loss: {val_loss:.4f}, val acc: {val_acc:.4f} lr:{param_lr}')

    torch.save(model.state_dict(), os.path.join(workspace_path, f'last_model.pth'))

    artifact.add_file(os.path.join(workspace_path, f'last_model.pth'))
    artifact.add_file(os.path.join(workspace_path, f'best_model.pth'))
    artifact.add_file(os.path.join(workspace_path, f'categories.json'))
    wandb.log_artifact(artifact)




def evaluation(model, val_loader, criteria, device, workspace_path, tconfig, categories):
    use_meta = tconfig.get('use_meta', True)

    routing_config = tconfig.get('routing_criterion', None)

    checkpoint_path = os.path.join(workspace_path, 'best_model.pth')
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    print(f'Loaded checkpoint : {checkpoint_path}')

    val_loss, val_acc, val_verbose = evaluate_epoch(0, val_loader, model, criteria, device, use_meta=use_meta)
    stats = calc_statistics(targets=val_verbose['targets'], predictions=val_verbose['predictions'], categories=categories, category_name='pathology')

    return val_loss, val_acc, val_verbose, stats




def extract_metadata_features(data_loader, categories):
    """
    Extract metadata features from the dataloader in the same format used by the DL model.
    Converts one-hot encoded metadata back to feature vectors suitable for tabular models.
    """
    all_features = []
    all_labels = []
    feature_names = []

    # Get feature names from categories (excluding pathology which is the target)
    meta_keys = [k for k in categories.keys() if k != 'pathology']

    for inputs, labels, meta, index in data_loader:
        batch_size = labels.size(0)

        for i in range(batch_size):
            sample_features = []
            for key in meta_keys:
                if key in meta:
                    # meta[key] is a one-hot encoded tensor, get the argmax as the feature value
                    feature_val = meta[key][i].argmax().item()
                    sample_features.append(feature_val)

            all_features.append(sample_features)
            all_labels.append(labels[i].item())

    # Build feature names list (only once)
    feature_names = meta_keys

    return np.array(all_features), np.array(all_labels), feature_names


def train_tabular_models(X_train, y_train, X_val, y_val, feature_names):
    """
    Train multiple tabular models and return their predictions and metrics.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB

    try:
        from xgboost import XGBClassifier
        has_xgboost = True
    except ImportError:
        has_xgboost = False
        print("XGBoost not installed, skipping...")

    try:
        from lightgbm import LGBMClassifier
        has_lightgbm = True
    except ImportError:
        has_lightgbm = False
        print("LightGBM not installed, skipping...")

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
    }

    if has_xgboost:
        models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')

    if has_lightgbm:
        models['LightGBM'] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)

            # Get predictions
            y_pred = model.predict(X_val)

            # Get probability predictions for AUC
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_val)
            else:
                y_proba = None

            results[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_proba': y_proba
            }

            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                results[name]['feature_importance'] = dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                results[name]['feature_importance'] = dict(zip(feature_names, np.abs(model.coef_[0])))

        except Exception as e:
            print(f"Error training {name}: {e}")

    return results


def calc_tabular_statistics(y_true, y_pred, y_proba, categories):
    """
    Calculate the same statistics as used in the DL model evaluation.
    """
    stats = {}
    stats['acc'] = accuracy_score(y_true, y_pred)
    stats['f1'] = f1_score(y_true, y_pred)
    stats['recall'] = recall_score(y_true, y_pred)

    # Calculate AUC if probabilities are available
    if y_proba is not None and y_proba.shape[1] >= 2:
        try:
            # Use probability of positive class (malignant)
            stats['auc'] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception as e:
            print(f"Could not calculate AUC: {e}")
            stats['auc'] = np.nan

    return stats


def generate_results_table(all_results, output_dir, dataset_name=None):
    """
    Generate CSV and LaTeX tables from results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build filename suffix from dataset name
    suffix = f"_{dataset_name}" if dataset_name else ""

    # Prepare data for table
    rows = []
    for model_name, stats in all_results.items():
        row = {
            'Model': model_name,
            'Accuracy': f"{stats['acc']*100:.1f}",
            'AUC': f"{stats.get('auc', np.nan)*100:.1f}" if not np.isnan(stats.get('auc', np.nan)) else 'N/A',
            'F1 score': f"{stats['f1']*100:.1f}",
            'Recall': f"{stats['recall']*100:.1f}"
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(output_dir, f'meta_evaluation_results{suffix}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

    # Generate LaTeX table
    tex_content = "\\begin{tabular}{lllll}\n"
    tex_content += "\\toprule\n"
    tex_content += "Model & Accuracy & AUC & F1 score & Recall \\\\\n"
    tex_content += "\\midrule\n"

    for _, row in df.iterrows():
        tex_content += f"{row['Model']} & {row['Accuracy']} & {row['AUC']} & {row['F1 score']} & {row['Recall']} \\\\\n"

    tex_content += "\\bottomrule\n"
    tex_content += "\\end{tabular}\n"

    tex_path = os.path.join(output_dir, f'meta_evaluation_results{suffix}.tex')
    with open(tex_path, 'w') as f:
        f.write(tex_content)
    print(f"LaTeX table saved to: {tex_path}")

    return df


def generate_feature_importance_table(all_results, feature_names, output_dir, dataset_name=None):
    """
    Generate feature importance table from models that support it.
    """
    importance_data = {}

    for model_name, result in all_results.items():
        if 'feature_importance' in result:
            importance_data[model_name] = result['feature_importance']

    if not importance_data:
        return None

    # Build filename suffix from dataset name
    suffix = f"_{dataset_name}" if dataset_name else ""

    # Create DataFrame
    df = pd.DataFrame(importance_data)
    df.index.name = 'Feature'

    # Save CSV
    csv_path = os.path.join(output_dir, f'feature_importance{suffix}.csv')
    df.to_csv(csv_path)
    print(f"Feature importance saved to: {csv_path}")

    return df


def meta_evaluation(train_loader, val_loader, criteria, device, workspace_path, tconfig, categories, dataset_name=None):
    """
    Train and evaluate tabular models on metadata features.
    Uses the same metadata that would be used in the DL model.
    Outputs results to CSV and TeX files in the paper directory.
    """
    print("\n" + "=" * 60)
    print("META EVALUATION: Training Tabular Models on Metadata")
    print("=" * 60)

    # Extract features from dataloaders
    print("\nExtracting metadata features from training set...")
    X_train, y_train, feature_names = extract_metadata_features(train_loader, categories)
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Features: {feature_names}")

    print("\nExtracting metadata features from validation set...")
    X_val, y_val, _ = extract_metadata_features(val_loader, categories)
    print(f"Validation set: {X_val.shape[0]} samples")

    # Print class distribution
    print(f"\nTraining class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Validation class distribution: {dict(zip(*np.unique(y_val, return_counts=True)))}")

    # Train models
    print("\n" + "-" * 40)
    print("Training tabular models...")
    print("-" * 40)
    model_results = train_tabular_models(X_train, y_train, X_val, y_val, feature_names)

    # Calculate statistics for each model
    print("\n" + "-" * 40)
    print("Evaluating models...")
    print("-" * 40)
    all_stats = {}
    for model_name, result in model_results.items():
        stats = calc_tabular_statistics(
            y_val,
            result['y_pred'],
            result['y_proba'],
            categories
        )
        all_stats[model_name] = stats

        # Also store feature importance if available
        if 'feature_importance' in result:
            all_stats[model_name]['feature_importance'] = result['feature_importance']

        print(f"{model_name}: Acc={stats['acc']*100:.1f}%, AUC={stats.get('auc', np.nan)*100:.1f}%, "
              f"F1={stats['f1']*100:.1f}%, Recall={stats['recall']*100:.1f}%")

    # Generate output tables
    print("\n" + "-" * 40)
    print("Generating output tables...")
    print("-" * 40)

    # Output to paper directory
    paper_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'paper')
    os.makedirs(paper_dir, exist_ok=True)

    # Generate main results table
    df_results = generate_results_table(all_stats, paper_dir, dataset_name=dataset_name)

    # Generate feature importance table
    df_importance = generate_feature_importance_table(model_results, feature_names, paper_dir, dataset_name=dataset_name)

    # Also save to workspace for reference
    workspace_meta_dir = os.path.join(workspace_path, 'meta_evaluation')
    os.makedirs(workspace_meta_dir, exist_ok=True)
    generate_results_table(all_stats, workspace_meta_dir, dataset_name=dataset_name)
    generate_feature_importance_table(model_results, feature_names, workspace_meta_dir, dataset_name=dataset_name)

    print("\n" + "=" * 60)
    print("META EVALUATION COMPLETE")
    print("=" * 60)

    return all_stats


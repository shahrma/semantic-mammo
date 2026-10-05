import torch.nn as nn
import torch

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

import pandas as pd
from torch.optim.lr_scheduler import _LRScheduler
from core.LDN_wrapper import collect_model_stats

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class RoutingLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        # This code does not create a new object.
        # It stores a reference (pointer) to the original model that was passed in
        self.model = model
        self.lambda_balance = None

    def get_routing_probs(self):
        all_probs, _ = collect_model_stats(self.model)
        return all_probs

    def get_lambda_balance(self):
        return self.lambda_balance

class UniformLoss(RoutingLoss):
    def __init__(self, model, config=None):
        super().__init__(model)
        if config is None:
            config = {}
        self.base_lambda_balance = config.get('lambda_balance', 0.5)
        self.lambda_last_epoch = config.get('lambda_last_epoch', None)
        self.lambda_step = config.get('lambda_step', 1)
        self.width = 2
        self.criterion = nn.MSELoss()

    def get_curr_lambda_balance(self, epoch):
        if self.lambda_last_epoch is None:
            lambda_balance = self.base_lambda_balance
        else:
            steps = epoch // self.lambda_step
            total_steps = self.lambda_last_epoch // self.lambda_step
            lambda_balance = max(self.base_lambda_balance * (1 - steps / total_steps), 0)

        lambda_balance = round(lambda_balance, 3)

        self.lambda_balance = lambda_balance
        return lambda_balance


    def forward(self, epoch):
        probs = self.get_routing_probs()
        if probs is None:
            return 0, None

        mean_probs = probs.mean(dim=0)
        num_routes = mean_probs.shape[-1]
        target_probs = torch.full_like(mean_probs, 1 / self.width)
        loss = self.criterion(mean_probs, target_probs)

        lambda_balance = self.get_curr_lambda_balance(epoch)

        balanced_loss = lambda_balance * loss

        return balanced_loss , probs

def produce_stats(all_probs, prefix = 'prob'):
    all_probs = torch.cat(all_probs)
    df = pd.DataFrame(all_probs.numpy(), columns=[f'{prefix}_{i}' for i in range(all_probs.shape[1])])
    stats_df = pd.DataFrame({
        'mean': df.mean(),
        'std': df.std()
    }).reset_index().rename(columns={'index': f'{prefix}'})

    return stats_df

def get_routing_criterion(model, config):
    criterion_type = config.get('criterion_type', 'UNI')
    if criterion_type == 'UNI':
        criterion = UniformLoss(model, config)
    else:
        criterion = None

    return criterion

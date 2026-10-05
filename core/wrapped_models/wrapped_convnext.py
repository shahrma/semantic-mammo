import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import fnmatch

from torchvision.models import convnext_base

from core.wrapped_models.meta_wrapper import BaseWrapper

class MetaLayerNorm(nn.Module):
    def __init__(self, original_layer, group_name=None, params=None, meta_dim=None, eps=1e-5):
        super().__init__()

        self.group_name = group_name
        self.params = params
        self.normalized_shape = original_layer.normalized_shape
        self.eps = eps
        self.layer = nn.LayerNorm(normalized_shape=original_layer.normalized_shape, eps=original_layer.eps, elementwise_affine=False)

        self.meta_dim = self.params.get('meta_dim',meta_dim)
        # add_unknown = self.params.get('add_unknown',False)
        # self.meta_dim = self.meta_dim+1 if add_unknown else self.meta_dim

        active_mode = self.params.get('active_mode', 'meta-fc')
        init_type = self.params.get('init_type', None)
        if active_mode == 'no_meta':
            self.gamma = nn.Parameter(original_layer.weight.clone())
            self.beta = nn.Parameter(original_layer.bias.clone())
        elif active_mode in ['mlp','shallow_nn'] :
            self.gamma_nn, self.beta_nn = self.create_hyper_network(in_features=self.meta_dim, out_features=self.normalized_shape[0],
                                                                    active_mode=active_mode, params=self.params)
            if init_type == 'biased':
                self.initialize_sequential(self.gamma_nn, original_layer.weight)
                self.initialize_sequential(self.beta_nn, original_layer.bias)
        else:
            raise f'MetaLayerNorm does not recognize - {active_mode}'
        self.meta = None

    def initialize_sequential(self, seq, target=None):
        std = target.std().item() / (10*target.abs().sum().item())

        with torch.no_grad():
            for layer in seq[:-1]:
                if isinstance(layer, nn.Linear):
                    layer.weight.copy_(torch.normal(mean=0.0, std=std, size=layer.weight.shape))
                    layer.bias.fill_(0.0)

            last_layer = seq[-1]
            if isinstance(last_layer, nn.Linear):
                last_layer.bias.copy_(target-1)
                last_layer.weight.copy_(torch.normal(mean=0.0, std=std, size=last_layer.weight.shape))


    def create_hyper_network(self, in_features, out_features, active_mode, params = None):
        params = dict() if params is None else params
        if active_mode == 'mlp':
            gamma_nn = nn.Sequential(nn.Linear(in_features, out_features))
            beta_nn = nn.Sequential(nn.Linear(in_features, out_features))
        elif active_mode == 'shallow_nn':
            hidden_dim = params.get('hidden_dim', 32)
            gamma_nn = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_features)
            )
            beta_nn = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_features)
            )
        else:
            raise f'MetaLayerNorm does not recognize - {active_mode}'

        return gamma_nn, beta_nn

    def set_meta(self, meta):
        self.meta = copy.deepcopy(meta)

    def forward_no_meta(self, x):
        gamma = self.gamma
        beta = self.beta
        x_norm = self.layer(x)
        out = x_norm * gamma + beta
        return out


    def forward(self, x):
        active_mode = self.params.get('active_mode', 'meta-fc')
        if active_mode == 'no_meta':
            return self.forward_no_meta(x)

        if self.meta is None:
            raise ValueError("meta is not set — call .set_meta(meta) before forward.")
        meta = self.meta[self.group_name].float()
        gamma = self.gamma_nn(meta)
        beta = self.beta_nn(meta)
        gamma = gamma[:, None, None, :]  # [B, 1, 1, C]
        beta = beta[:, None, None, :]  # [B, 1, 1, C]

        x_norm = self.layer(x)
        out = x_norm * gamma + beta

        return out



def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters(recurse=True):
        p.requires_grad = flag
from torchvision.models.convnext import CNBlock

def unfreeze_last_cnblocks(model: nn.Module, n_blocks: int, unfreeze_classifier=True):
    """
    Unfreeze only the last `n_blocks` CNBlocks in ConvNeXt.
    n_blocks ∈ {1, 2, 3}
    """

    assert 1 <= n_blocks <= 3, "ConvNeXt last stage has 3 CNBlocks"

    # 1) Freeze everything
    set_requires_grad(model, False)

    # 2) Optionally unfreeze classifier
    if unfreeze_classifier and hasattr(model, "classifier"):
        set_requires_grad(model.classifier, True)

    # 3) Get last ConvNeXt stage (features[7])
    last_stage = model.features[7]

    # 4) Select last N CNBlocks
    cnblocks = [m for m in last_stage if isinstance(m, CNBlock)]
    for block in cnblocks[-n_blocks:]:
        set_requires_grad(block, True)

def freeze_layers(model: nn.Module):
    # 1) freeze everything
    set_requires_grad(model, False)

    # 2) unfreeze classifier
    if hasattr(model, "classifier"):
        set_requires_grad(model.classifier, True)

    # 3) unfreeze LayerNorms (including LayerNorm2d)
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):  # GroupNorm optional; remove if not needed
            set_requires_grad(m, True)
        # torchvision uses LayerNorm2d class; safest is to detect by name too
        if m.__class__.__name__ == "LayerNorm2d":
            set_requires_grad(m, True)
        # your custom MetaLayerNorm: unfreeze it fully
        if m.__class__.__name__ == "MetaLayerNorm":
            set_requires_grad(m, True)

    # 4) unfreeze all CNBlocks
    unfreeze_last_cnblocks(model, n_blocks=1)



def print_layer_trainability(model: nn.Module):
    """
    Prints each module and whether it has any trainable parameters.
    """
    print(f"{'Layer name':60s} | Trainable")
    print("-" * 75)

    for name, module in model.named_modules():
        params = list(module.parameters(recurse=False))

        if not params:
            status = "—"  # no parameters in this module
        else:
            trainable = any(p.requires_grad for p in params)
            status = "YES" if trainable else "NO"

        print(f"{name:60s} | {status}")

#
# def list_cnblocks(model: nn.Module):
#     blocks = []
#     for name, m in model.named_modules():
#         if isinstance(m, CNBlock):
#             blocks.append(name)
#     return blocks


class WrappedConvNeXt(BaseWrapper):
    def __init__(self, weights=None, replacing_list=None, num_classes=None, categories=None):
        super().__init__(replaced_type=nn.LayerNorm,
                         replacement_type=MetaLayerNorm,)

        self.model = convnext_base(weights=weights)
        replaced_list = self.wrap_layers(replacing_list, categories)
        print(replacing_list)
        print(replaced_list)

        self.model.classifier[2] = torch.nn.Linear(self.model.classifier[2].in_features, num_classes)
        # freeze_layers(self.model)
        #
        # trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # total = sum(p.numel() for p in self.model.parameters())
        # print(f"Trainable params: {trainable:,} / {total:,}")

    def forward(self, x, meta):
        super().set_meta(meta)

        out = self.model(x)

        # outputs = [{'group': 'main','name':'main', 'predicted':out}]

        return out


if __name__ == '__main__':
    replacing_list = {'breast density':  {'layers': ['features.3.1*'], 'params': {'meta_dim': 6,'active_mode':'no_meta'}},
                      'mass_shape_standard': {'layers': ['features.5.2*'], 'params': {'meta_dim': 4,'active_mode':'no_meta'}}}

    num_classes = 2


    model = WrappedConvNeXt(replacing_list=replacing_list, num_classes=num_classes)


    batch_size = 8
    x = torch.randn(batch_size, 3, 224, 224)

    meta = {}
    for group_name, item in replacing_list.items():
        meta[group_name] = torch.randn(batch_size, item['params']['meta_dim'])

    output = model(x, meta)
    print(output)


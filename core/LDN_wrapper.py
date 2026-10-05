import copy

import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F
from numpy import random
import fnmatch
import torch.fx as fx

from collections import defaultdict


def collect_model_stats(model):
    routing_probs = []
    routing_scalars = []
    for module in model.modules():
        if isinstance(module, LDNConv) and module.logits is not None:
            p = torch.sigmoid(module.logits)
            routing_probs.append(p)
            routing_scalars.append(torch.cat([module.scalar1.detach().cpu().unsqueeze(0),
                                              module.scalar2.detach().cpu().unsqueeze(0)]))

    all_scalars = None
    if len(routing_scalars) > 0:
        all_scalars = torch.stack(routing_scalars, dim=0)

    all_probs = None
    if len(routing_probs) > 0:
        all_probs = torch.cat(routing_probs, dim=1)

    return all_probs, all_scalars

class LDNConv(nn.Module):
    def __init__(self, conv_layer, noise_type):
        super(LDNConv, self).__init__()
        self.route_decision_soft = None
        self.conv = conv_layer
        self.scalar1 = nn.Parameter(torch.tensor(0.1))
        self.scalar2 = nn.Parameter(torch.tensor(0.9))
        self.fc = nn.Linear(conv_layer.in_channels, 1)
        self.logits = None
        self.route_decision_binary = None
        self.noise_type = noise_type
        self.noise_power = 1

    def forward(self, x, mode='binary'):
        gap = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        self.logits = self.fc(gap)

        if self.training:
            if self.noise_type == 'gumbel':
                noise = -torch.empty_like(self.logits).exponential_().log() * self.noise_power
            else:
                noise = self.noise_power*torch.randn_like(self.logits)

            g_omega = self.logits + noise
            gr = torch.clamp(1.2 * torch.sigmoid(g_omega) - 0.1, 0, 1)
            gb = (gr > 0.5).float()

            rand_mask = (torch.rand_like(gb) > 0.5).float()
            decision = gb * rand_mask + gr * (1 - rand_mask)

            route_decision = decision + gr - gr.detach()
        else:
            route_decision = (torch.sigmoid(self.logits) > 0.5).float()

        scaled_weight1 = self.conv.weight * self.scalar1
        scaled_weight2 = self.conv.weight * self.scalar2

        out1 = F.conv2d(
            x, scaled_weight1, self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups
        )
        out2 = F.conv2d(
            x, scaled_weight2, self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups
        )

        if self.training:
            if random.random() < 0.5:
                # print("soft")
                self.route_decision_soft = route_decision.view(-1, 1, 1, 1)
                return out1 * self.route_decision_soft + out2 * (1 - self.route_decision_soft)
            else:
                # print("hard")
                self.route_decision_binary = (route_decision > 0.5).float().view(-1, 1, 1, 1)
                return out1 * self.route_decision_binary + out2 * (1 - self.route_decision_binary)
        else:
            # print("hard")
            self.route_decision_binary = (route_decision > 0.5).float().view(-1, 1, 1, 1)
            return out1 * self.route_decision_binary + out2 * (1 - self.route_decision_binary)


def replace_layer(model, layer_name, new_layer):
    """Replace a layer in the model given its full name."""
    components = layer_name.split('.')
    module = model

    # Traverse the path to the last module
    for comp in components[:-1]:
        module = getattr(module, comp)

    # Replace the target layer
    setattr(module, components[-1], new_layer)



def change_conv_depth_hierarchical(model, target_layer_name, new_out_channels, factor=None):
    # Trace the model
    traced = fx.symbolic_trace(model)

    # Build mapping from node targets to module names and vice versa
    name_to_module = dict(model.named_modules())

    # Build node → users map
    users_map = defaultdict(list)
    for node in traced.graph.nodes:
        for user in node.users:
            users_map[node].append(user)

    # Find the target node
    target_node = None
    for node in traced.graph.nodes:
        if node.op == 'call_module' and target_layer_name in node.target:
            target_node = node
            break
    assert target_node is not None, f"Layer '{target_layer_name}' not found"

    # Get old module and replace it
    print('')
    print('----------------------------------')
    print(f'Replacing --- {target_node.target}')
    print('----------------------------------')
    old_conv = name_to_module[target_node.target]
    new_out_channels = new_out_channels if factor is None else round(old_conv.out_channels / factor)
    new_conv = nn.Conv2d(
        in_channels=old_conv.in_channels,
        out_channels=new_out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )
    replace_layer(model, target_node.target, new_conv)
    name_to_module[target_node.target] = new_conv

    # Visited nodes to avoid infinite loops
    visited = set()

    def update_downstream_users(node, out_channels):
        if node in visited:
            return
        visited.add(node)

        for user in users_map[node]:
            if user.op == 'call_module':
                mod = name_to_module.get(user.target, None)
                print(f'user --- {user.target}')
                if isinstance(mod, nn.Conv2d):
                    new_mod = nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=mod.out_channels,
                        kernel_size=mod.kernel_size,
                        stride=mod.stride,
                        padding=mod.padding,
                        bias=mod.bias is not None
                    )
                    replace_layer(model, user.target, new_mod)
                    name_to_module[user.target] = new_mod
                elif isinstance(mod, nn.Linear):
                    new_mod = nn.Linear(
                        in_features=out_channels,
                        out_features=mod.out_features,
                        bias=mod.bias is not None
                    )
                    replace_layer(model, user.target, new_mod)
                    name_to_module[user.target] = new_mod
                elif isinstance(mod, nn.BatchNorm2d):
                    new_bn = nn.BatchNorm2d(out_channels)
                    replace_layer(model, user.target, new_bn)
                    name_to_module[user.target] = new_bn
                    update_downstream_users(user, out_channels)
                else:
                    # Add more types if needed (e.g., Linear, ReLU, etc.)
                    update_downstream_users(user, out_channels)
            else:
                update_downstream_users(user, out_channels)

    # Start recursive update from the modified layer
    update_downstream_users(target_node, new_out_channels)

    return model



def depth_replacer(model, layers=None, new_out_channels=32,factor=None):
    layers_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layers_to_replace.append(name)

    if layers is not None:
        matched_layers = []
        for  pattern in layers:
            matched_layers.extend(fnmatch.filter(layers_to_replace, pattern))
        layers_to_replace = sorted(list(set(matched_layers)))

    for layer_name in layers_to_replace:
        model = change_conv_depth_hierarchical(model, layer_name, new_out_channels=new_out_channels, factor=factor)
        # for name, module in model.named_modules():
        #     if isinstance(module, nn.Conv2d):
        #         if name in layers_to_replace:

    return model


def LDN_warpping(model, layers=None, noise_type='gaussian'):
    layers_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layers_to_replace.append(name)

    if layers is not None:
        matched_layers = []
        for  pattern in layers:
            matched_layers.extend(fnmatch.filter(layers_to_replace, pattern))
        layers_to_replace = sorted(list(set(matched_layers)))

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if name in layers_to_replace:
                replace_layer(model, name, LDNConv(module, noise_type=noise_type))
                print(f'{name:<40} {str("-- replaced --"):<20} {str(module):<80}')
            else:
                print(f'{name:<40} {str("              "):<20} {str(module):<80}')

    return model


if __name__ == '__main__':
    import torchvision.models as models
    from torchinfo import summary
    from copy import deepcopy
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    print('\n\n\nReplace pattern layers')
    model_specific = depth_replacer(deepcopy(model),  layers=[f'layer{i}.*.conv*' for i in [2,3,4]], factor=8)
    model_specific = depth_replacer(deepcopy(model_specific),  layers=[f'layer{i}.*.downsample.*' for i in [2,3,4]], factor=8)
    model_specific = LDN_warpping(deepcopy(model_specific), layers=['layer*.*.conv*'])
    #
    #
    # model_specific = depth_replacer(deepcopy(model),  layers=['layer2.*.conv*'], new_out_channels=32)
    # model_specific = depth_replacer(deepcopy(model_specific),  layers=['layer2.*.downsample.*'], new_out_channels=32)
    # model_specific = LDN_warpping(deepcopy(model_specific), layers=['layer2.*.conv2'])

    # print('\n\n\nReplace all conv layers')
    # model_all = LDN_warpping(deepcopy(model), layers=None)
    #
    # print('\n\n\nReplace specific layers')
    # model_specific2 = LDN_warpping(deepcopy(model), layers=['layer2.1.conv2',
    #                                     'layer3.0.downsample.0',
    #                                     'layer4.0.conv2'])
    # print('\n\n\nDo not replace any layer')
    # model_orig = LDN_warpping(deepcopy(model), layers=[])

    # Create a dummy input tensor with shape (1, 3, 32, 32)
    device = next(model_specific.parameters()).device

    dummy_input = torch.randn(4, 3, 32, 32).to(device)
    model_specific(dummy_input)

    # summary(model, (1, 3, 32, 32))
    # summary(model_all, (1, 3, 32, 32))
    summary(model_specific, (1, 3, 32, 32))
    summary(model_orig, (1, 3, 32, 32))
    summary(model_specific2, (1, 3, 32, 32))

    pass
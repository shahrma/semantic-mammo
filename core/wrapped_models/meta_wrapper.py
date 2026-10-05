import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import fnmatch



class BaseWrapper(nn.Module):
    def __init__(self, replaced_type, replacement_type, **replacement_kwargs):
        super().__init__()
        self.replaced_type = replaced_type
        self.replacement_type = replacement_type
        self.replacement_kwargs = replacement_kwargs

        self.model = None
        self.replaced_layers = {}

    def wrap_layers(self, replacing_list, categories=None):
        replacing_list = copy.deepcopy(replacing_list)
        for group_name, item in replacing_list.items():
            meta_dim = len(categories[group_name])
            replaced_layers = self.wrapping(group_name=group_name, layers=item['layers'], params=item['params'], meta_dim=meta_dim)
            self.replaced_layers[group_name] = item.update({'replaced':replaced_layers})
        return replacing_list

    def set_meta(self, meta=None):
        for module in self.model.modules():
            if isinstance(module, self.replacement_type):
                module.set_meta(meta)

    def replace_layer(self, model, layer_name, new_layer):
        """Replace a layer in the model given its full name."""
        components = layer_name.split('.')
        module = model

        # Traverse the path to the last module
        for comp in components[:-1]:
            module = getattr(module, comp)

        # Replace the target layer
        setattr(module, components[-1], new_layer)

    def wrapping(self, group_name=None, layers=None, params=None, meta_dim=None ):
        layers_to_replace = []
        for name, module in self.model.named_modules():
            if isinstance(module, self.replaced_type):
                layers_to_replace.append(name)

        if layers is not None:
            matched_layers = []
            for  pattern in layers:
                matched_layers.extend(fnmatch.filter(layers_to_replace, pattern))
            layers_to_replace = sorted(list(set(matched_layers)))

        replaced_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, self.replaced_type):
                if name in layers_to_replace:
                    self.replace_layer(self.model, name, self.replacement_type(module,group_name,params,meta_dim=meta_dim))
                    replaced_layers.append(name)
                    print(f'{name:<40} {str("-- replaced --"):<20} {str(module):<80}')
                else:
                    print(f'{name:<40} {str("              "):<20} {str(module):<80}')

        return replaced_layers

class ExtendedWrapper(BaseWrapper):
    def __init__(self, replaced_type, replacement_type, **replacement_kwargs):
        # call parent constructor
        super().__init__(replaced_type, replacement_type, **replacement_kwargs)

    def get_predicted_meta(self):
        output = []
        # for module in self.model.modules():
        for name, module in self.model.named_modules():
            if isinstance(module, self.replacement_type):
                pmeta = module.get_predicted_meta()
                pmeta.update({'name':name})
                output.append(pmeta)

        return output
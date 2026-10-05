import copy

import yaml

def update_nested_dict(base_dict, new_dict):
    for key, value in new_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            # If the value is a dictionary and exists in both, update recursively
            update_nested_dict(base_dict[key], value)
        else:
            # Otherwise, replace or add the value
            base_dict[key] = value

def load_yaml_config(filename):
    with open(filename, "r") as file:
        config = yaml.safe_load(file)

    base_config_file = config.pop('base_config', None)

    if base_config_file is not None:
        base_config = load_yaml_config(base_config_file)
        update_nested_dict(base_config, config)
        config = copy.deepcopy(base_config)

    return config

def set_nested_value(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

import ast
def set_nested_value_from_str_path(d, path_str, value):
    keys = path_str.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    if isinstance(d[keys[-1]], list):
        d[keys[-1]] = ast.literal_eval(value)
    else:
        d[keys[-1]] = type(d[keys[-1]])(value)

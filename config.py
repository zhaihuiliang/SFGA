import os
import argparse
from yacs.config import CfgNode as CN
from itertools import product

def print_cfg(cfg): 
    print("Train Config:")
    for key, value in cfg.items():
        if not isinstance(value, CN):
            print(f"  {key}: {value}")

    print("Dataset Config:")
    for key, value in cfg.dataset.items():
        print(f"  {key}: {value}")

    print("gnn Config:")
    for key, value in cfg.gnn.items():
        print(f"  {key}: {value}")

    print(f"{cfg.model_name} Config:")
    for key, value in cfg[cfg.model_name].items():
        print(f"  {key}: {value}")

def set_cfg(cfg):
    # ------------------------------------------------------------------------ #
    # Basic settings
    # ------------------------------------------------------------------------ #
    cfg.dataset_name = ''
    cfg.model_name = ''
    cfg.gpu_num = 0
    cfg.seed = [0]
    cfg.lr = 1e-3
    cfg.weight_decay = 1e-4
    cfg.use_batch = False
    cfg.dataset = CN()
    cfg.gnn = CN()
    cfg.sfga = CN()

    cfg.dataset.root = './data/'
    cfg.dataset.sc = 1
    cfg.dataset.sparse = False

    cfg.gnn.hidden_dim = 64
    cfg.gnn.layers = 2
    cfg.gnn.dropout = 0.1
    cfg.gnn.isBias = True
    cfg.gnn.activation = 'relu'

    # ------------------------------------------------------------------------ #
    # SFGA settings
    # ------------------------------------------------------------------------ #
    cfg.sfga.emb_dim = 0
    cfg.sfga.alpha = 0.0
    cfg.sfga.beta = 0.0
    cfg.sfga.decolayer = 0
    cfg.sfga.inner_epochs = 0

    return cfg

def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER, help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from command line
    cfg.merge_from_list(args.opts)

    #Update from config file
    config_path = f"yamls/sfga.yaml"
    if os.path.isfile(config_path):
        model_cfg = CN.load_cfg(open(config_path, 'r'))
    else:
        raise FileNotFoundError(f"Model-specific config file not found: {config_path}")
    model_dataset_cfg = getattr(model_cfg, cfg.dataset_name.lower(), None)
    if model_dataset_cfg is None:
        raise KeyError(f"Dataset section '{cfg.dataset_name.lower()}' not found in the model's config file")

    if cfg.dataset_name.lower() in ['imdb_injected']:
        cfg.epochs = 10
    elif cfg.dataset_name.lower() in ['freebase_injected']:
        cfg.epochs = 10
    elif cfg.dataset_name.lower() in ['amazon_fraud']:
        cfg.epochs = 10
    elif cfg.dataset_name.lower() in ['yelpchi_fraud']:
        cfg.epochs = 2
    else:
        raise KeyError(f"Dataset section '{cfg.dataset_name.lower()}' not right when setting epochs")

    ### Grid Search
    # Flatten the nested dictionary
    def flatten_dict(d, parent_key='', sep=','):
        items = []
        for k, v in d.items():
            new_key = f'{parent_key}{sep}{k}' if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    # Flatten the nested dictionary
    flat_model_dataset_cfg = flatten_dict(model_dataset_cfg)

    # Hyperparameter grid search
    grid_search_params = []
    for key, values in flat_model_dataset_cfg.items():
        if isinstance(values, list):
            grid_search_params.append((key,values))
        else:
            grid_search_params.append((key, [values]))

    # Generate combinations for grid search
    grid_combinations = product(*[values for _, values in grid_search_params])

    # Perform grid search
    cfg_list = []
    for combination in grid_combinations:
        cfg_copy = cfg.clone()
        # Update the config directly
        for param, value in zip([param for param, _ in grid_search_params], combination):
            keys = param.split(',')
            current_dict = cfg_copy
            for key in keys[:-1]:
                current_dict = current_dict.setdefault(key, {})
            current_dict[keys[-1]] = value
        cfg_list.append(cfg_copy)
    return cfg_list

"""
    Global variable
"""
cfg = set_cfg(CN())
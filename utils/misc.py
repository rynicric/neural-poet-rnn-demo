from types import SimpleNamespace
import random
import yaml
import torch
import numpy as np


def config_hook(config_file, **kwargs):
    ''' Configuration processing.

    Args:
      - config_file (str): yaml config file path
    '''
    class NestedNamespace(SimpleNamespace):
        def __init__(self, dictionary, **kwargs):
            super().__init__(**kwargs)
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    self.__setattr__(key, NestedNamespace(value))
                else:
                    self.__setattr__(key, value)

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        config = NestedNamespace(config)

    for k, v in kwargs.items():
        setattr(config, k, v)

    # Fix the terminal input problem about the array type on `device`.
    if hasattr(config, 'device') and type(config.device) == tuple:
        setattr(config, 'device', ','.join(map(str, config.device)))

    return config


def keep_seed(seed):
    ''' Specify random seeds to ensure reproducible experiments.

    Args:
      - seed (num): random seed
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_cache(model_state_dict, cache_state_dict):
    ''' Load the model cache correctly.

    Args:
      - model_state_dict (dict): model state dict from raw net
      - cache_state_dict (dict): cache state dict from weight file

    Returns:
      - (dict): processed model state dict
    '''
    filtered = dict()
    unloadkey = []
    for k, v in cache_state_dict.items():
        if k in model_state_dict and model_state_dict[k].shape == v.shape:
            filtered[k] = v
        else:
            unloadkey.append(k)
        model_state_dict.update(filtered)
        print('update weights:', f'{len(filtered)}/{len(cache_state_dict)}')
        print('unload keys:', *unloadkey, sep='\n    ')
        return model_state_dict

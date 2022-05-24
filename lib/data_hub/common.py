
# -- python imports --
import numpy as np
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
from torch.utils.data import DataLoader

def optional(pydict,field,default):
    if pydict is None: return default
    if field in pydict.keys(): return pydict[field]
    else: return default

def get_loaders(cfg,data):
    loader = edict()
    loader.tr = DataLoader(data.tr)
    loader.val = DataLoader(data.val)
    loader.te = DataLoader(data.te)
    return loader


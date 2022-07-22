
# -- python imports --
import numpy as np
from easydict import EasyDict as edict

# -- pytorch imports --
import torch as th
from torch.utils.data import DataLoader

def optional(pydict,field,default):
    if pydict is None: return default
    if field in pydict.keys(): return pydict[field]
    else: return default

def collate_dict(batch,all_dim0=False):
    fbatch = {}
    for sample in batch:
        keys = sample.keys()
        for key,elem in sample.items():
            if not (key in fbatch): fbatch[key] = []
            fbatch[key].append(elem)

    skip_fields = ["rng_state"]
    for key in fbatch:
        if key in skip_fields: continue
        fbatch[key] = th.stack(fbatch[key])

    return fbatch

def get_loaders(cfg,data,batch_size):

    # -- results --
    loader = edict()

    # -- args --
    kwargs = edict()
    kwargs['batch_size'] = batch_size
    kwargs['collate_fn'] = collate_dict
    kwargs['num_workers'] = optional(cfg,'num_workers',2)
    kwargs['persistent_workers'] = optional(cfg,'persistent_workers',False)

    # -- train and val --
    if ("tr" in data) and not(data.tr is None):
        loader.tr = DataLoader(data.tr,**kwargs)
    if ("val" in data) and not(data.val is None):
        loader.val = DataLoader(data.val,**kwargs)

    # -- test --
    kwargs['batch_size'] = 1
    if ("te" in data) and not(data.te is None):
        loader.te = DataLoader(data.te,**kwargs)

    return loader

def get_isize(isize):
    """
    Convert 96_96 -> [96,96]
    """
    if not(isize is None) and (isize != "none"):
        isize = [int(x) for x in isize.split("_")]
    return isize


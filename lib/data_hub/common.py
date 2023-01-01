
# -- python imports --
import copy
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
    kwargs['collate_fn'] = collate_dict
    kwargs['num_workers'] = optional(cfg,'num_workers',2)
    kwargs['persistent_workers'] = optional(cfg,'persistent_workers',False)

    # -- train and val --
    if ("tr" in data) and not(data.tr is None):
        kwargs_tr = copy.deepcopy(kwargs)
        kwargs_tr['shuffle'] = optional(cfg,'rand_order_tr',True)
        kwargs_tr['batch_size'] = batch_size.tr
        loader.tr = DataLoader(data.tr,**kwargs_tr)
    if ("val" in data) and not(data.val is None):
        kwargs_val = copy.deepcopy(kwargs)
        kwargs_val['shuffle'] = optional(cfg,'rand_order_val',False)
        kwargs_val['batch_size'] = batch_size.val
        loader.val = DataLoader(data.val,**kwargs_val)

    # -- test --
    if ("te" in data) and not(data.te is None):
        kwargs_te = copy.deepcopy(kwargs)
        kwargs_te['shuffle'] = optional(cfg,'rand_order_te',False)
        kwargs_te['batch_size'] = batch_size.te
        loader.te = DataLoader(data.te,**kwargs_te)

    return loader

def get_isize(isize):
    """
    Convert 96_96 -> [96,96]
    """
    if not(isize is None) and (isize != "none"):
        isize = [int(x) for x in isize.split("_")]
    return isize

def filter_nframes(data_sub,vid_name,frame_start=-1,nframes=0):
    if nframes > 0: frame_end = frame_start + nframes - 1
    else: frame_end = -1
    return filter_eframe(data_sub,vid_name,frame_start,frame_end)

def filter_subseq(data_sub,vid_name,frame_start=-1,frame_end=-1):
    return filter_eframe(data_sub,vid_name,frame_start,frame_end)

def filter_eframe(data_sub,vid_name,frame_start=-1,frame_end=-1):

    """
    Filter a specific subsequence based on video name and frame indices.

    data_sub = data[subset]
    """

    # -- get initial indices --
    groups = data_sub.groups
    indices = [i for i,g in enumerate(groups) if (vid_name == g.split(":")[0])]

    # -- optional filter --
    if frame_start >= 0 and frame_end >= 0:
        def fbnds(fnums,lb,ub): return (lb <= np.min(fnums)) and (ub >= np.max(fnums))
        indices = [i for i in indices if fbnds(data_sub.paths['fnums'][groups[i]],
                                               frame_start,frame_end)]
    return indices



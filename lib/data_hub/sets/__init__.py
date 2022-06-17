
# -- rgb videos: (noisy_vid,clean_vid) pairs --
from . import toy64
from . import toy
from . import iphone
from . import set8
from . import sid

# -- real: (noisy_1,...,noisy_N,clean_frame) sets --
from . import submillilux

# -- single: (noisy_frame,clean_frame) sets --
from . import bsd68
from . import div2k

from data_hub.common import optional

def load(cfg):
    dname = optional(cfg,"dname","toy64")
    set_loaders = {"toy64":toy64,"toy":toy,"iphone":iphone,
                   "set8":set8,"sid":sid,"submillilux":submillilux,
                   "bsd68":bsd68,"div2k":div2k}
    dnames = list(set_loaders.keys())
    if not(dname in dnames):
        print("Options: ",dnames)
        raise ValueError(f"Uknown dname [{dname}].")
    data,loaders = set_loaders[dname].load(cfg)
    return data,loaders

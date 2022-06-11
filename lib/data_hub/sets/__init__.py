from . import toy64
from . import toy
from . import iphone
from . import set8
from . import sid
from . import submillilux
from data_hub.common import optional

def load(cfg):
    dname = optional(cfg,"dname","idk")
    set_loaders = {"toy64":toy64,"toy":toy,"iphone":iphone,
               "set8":set8,"sid":sid,"submillilux":submillilux}
    dnames = list(set_loaders.keys())
    if not(dname in dnames):
        raise ValueError(f"Uknown dname [{dname}].")
    data,loaders = set_loaders[dname].load(cfg)
    return data,loaders

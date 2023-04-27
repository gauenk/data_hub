
# -- rgb videos: (noisy_vid,clean_vid) pairs --
from . import toy64
from . import toy
from . import iphone
from . import set8
from . import sid
from . import sidd
from . import davis
from . import gopro
from . import iphone_s2023

# -- video + segmentation --
from . import cityscapes
from . import youtube_voc

# -- real: (noisy_1,...,noisy_N,clean_frame) sets --
from . import submillilux
from . import sird

# -- single: (noisy_frame,clean_frame) sets --
from . import bsd68
from . import div2k
from . import urban100
from . import set12

# -- segmentation --
from . import coco

# -- misc --
from data_hub.common import optional

def load(cfg):
    dname = optional(cfg,"dname","toy64")
    set_loaders = {"toy64":toy64,
                   "toy":toy,
                   "iphone":iphone,
                   "iphone_s2023":iphone_s2023,
                   "set8":set8,
                   "sid":sid,
                   "coco":coco.coco,
                   "cityscapes":cityscapes.cityscapes,
                   "youtube":youtube_voc.youtube_voc,
                   "youtube_voc":youtube_voc.youtube_voc,
                   "submillilux":submillilux,
                   "submillilux_real":submillilux.real,
                   "submillilux_paired":submillilux.paired,
                   "bsd68":bsd68,
                   "urban100":urban100,
                   "set12":set12,
                   "div2k":div2k,
                   "sidd_rgb_medium":sidd.rgb_medium,
                   "sidd_rgb_medium_cropped":sidd.rgb_medium_cropped,
                   "sidd":sidd.rgb_val,
                   "sidd_raw":sidd.raw_val,
                   "sidd_rgb":sidd.rgb_val,
                   "sidd_rgb_val":sidd.rgb_val,
                   "sidd_rgb_bench":sidd.rgb_bench,
                   "sidd_bench_full":sidd.bench_full,
                   "davis":davis,
                   "davis_cropped":davis.davis_cropped,
                   "gopro":gopro.gopro,
                   "gopro_cropped":gopro.gopro_cropped,
                   "sird":sird}
    dnames = list(set_loaders.keys())
    if not(dname in dnames):
        print("Options: ",dnames)
        raise ValueError(f"Uknown dname [{dname}].")
    data,loaders = set_loaders[dname].load(cfg)
    return data,loaders

"""
DAVIS dataset

"""

# -- python imports --
import pdb
import numpy as np
from pathlib import Path
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- pytorch imports --
import torch as th
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as tvF
from torchvision.transforms.functional import center_crop

# -- project imports --
from data_hub.common import get_loaders,optional,get_isize
from data_hub.transforms import get_noise_transform,noise_from_cfg
from data_hub.reproduce import RandomOnce,get_random_state,enumerate_indices
from data_hub.cropping import crop_vid
from data_hub.opt_parsing import parse_cfg
# apply_sobel_filter,sample_sobel_region,sample_rand_region

# -- local imports --
from .paths import CROPPED_BASE as BASE
from .paths import IMAGE_SETS
from .reader_cropped import read_names,read_data

# -- augmentations init --
import random
from data_hub.augmentations import get_scale_augs,get_flippy_augs

class DAVISCropped():

    def __init__(self,iroot,sroot,split,noise_info,
                 nsamples=0,nframes=0,fstride=1,isize=None,
                 bw=False,cropmode="coords",rand_order=False,
                 index_skip=1,flippy_augs=None,scale_augs=None):

        # -- set init params --
        self.iroot = iroot
        self.sroot = sroot
        self.split = split
        self.nframes = nframes
        self.isize = isize
        self.bw = bw
        self.rand_order = rand_order
        self.index_skip = index_skip

        # -- manage cropping --
        isize_is_none = isize is None or isize == "none"
        self.crop = isize
        self.cropmode = cropmode if not(isize_is_none) else "none"
        self.region_temp = None
        if not(isize_is_none):
            self.region_temp = "%d_%d_%d" % (nframes,isize[0],isize[1])

        # -- create transforms --
        self.noise_trans = get_noise_transform(noise_info,noise_only=True)

        # -- load paths --
        self.names = read_names(iroot,sroot,self.nframes,self.split,ext="jpg")
        self.groups = sorted(self.names)
        # self.paths = read_files(iroot,sroot,split,nframes,fstride,ext="jpg")
        # self.groups = sorted(list(self.paths['images'].keys()))

        # -- limit num of samples --
        self.indices = enumerate_indices(len(self.names),nsamples,
                                         rand_order,index_skip)
        self.nsamples = len(self.indices)

        # -- augmentations --
        self.flippy_augs = flippy_augs
        self.nflip_augs = 0 if flippy_augs is None else len(flippy_augs)
        self.scale_augs = scale_augs
        self.nscale_augs = 0 if scale_augs is None else len(scale_augs)

        # -- repro --
        self.noise_once = optional(noise_info,"sim_once",False)
        # self.fixRandNoise_1 = RandomOnce(self.noise_once,self.nsamples)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        # -- get random state --
        rng_state = None#get_random_state()

        # -- indices --
        image_index = self.indices[index]
        group = self.groups[image_index]

        # -- load burst --
        vid_name = self.names[image_index]
        clean,frame_nums = read_data(vid_name,self.iroot,self.nframes,self.bw)

        # -- augmentations --
        if self.nscale_augs > 0:
            aug_idx = random.randint(0,self.nscale_augs-1)
            trans_fxn = self.scale_augs[aug_idx]
            clean = trans_fxn(clean)
        if self.nflip_augs > 0:
            aug_idx = random.randint(0,self.nflip_augs-1)
            trans_fxn = self.flippy_augs[aug_idx]
            clean = trans_fxn(clean)

        # -- cropping --
        region = th.IntTensor([])
        use_region = "region" in self.cropmode or "coords" in self.cropmode
        if use_region:
            region = crop_vid(clean,self.cropmode,self.isize,self.region_temp)
        else:
            clean = crop_vid(clean,self.cropmode,self.isize,self.region_temp)

        # -- get noise --
        # with self.fixRandNoise_1.set_state(index):
        noisy = self.noise_trans(clean)

        # -- manage flow and output --
        index_th = th.IntTensor([image_index])

        return {'noisy':noisy,'clean':clean,'index':index_th,
                'fnums':frame_nums,'region':region,'rng_state':rng_state}

#
# Loading the datasets in a project
#

def get_davis_dataset(cfg):
    return load(cfg)


def load(cfg):

    #
    # -- extract --
    #

    # -- noise and dyanmics --
    noise_info = noise_from_cfg(cfg)

    # -- field names and defaults --
    modes = ['tr','val','te']
    fields = {"batch_size":1,
              "nsamples":-1,
              "isize":None,
              "fstride":1,
              "nframes":0,
              "fskip":1,
              "bw":False,
              "index_skip":1,
              "rand_order":False,
              "cropmode":"rand",
              "num_workers":2}
    p = parse_cfg(cfg,modes,fields)

    # -- augmentations --
    aug_flips = optional(cfg,"aug_training_flips",False)
    flippy_augs = get_flippy_augs() if aug_flips else None
    aug_scales = optional(cfg,"aug_training_scales",None)
    scale_augs = get_scale_augs(aug_scales)

    # -- setup paths --
    iroot = BASE
    sroot = IMAGE_SETS

    # -- create objcs --
    data = edict()
    data.tr = DAVISCropped(iroot,sroot,"train",noise_info,p.nsamples.tr,
                    p.nframes.tr,p.fstride.tr,p.isize.tr,p.bw.tr,p.cropmode.tr,
                    p.rand_order.tr,p.index_skip.tr,flippy_augs,scale_augs)
    data.val = DAVISCropped(iroot,sroot,"val",noise_info,p.nsamples.val,
                     p.nframes.val,p.fstride.val,p.isize.val,p.bw.val,p.cropmode.tr,
                     p.rand_order.val,p.index_skip.val)
    loader = get_loaders(cfg,data,p.batch_size)

    return data,loader


"""
Submillilux dataset

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

# -- project imports --
from data_hub.common import get_loaders,optional,get_isize
from data_hub.transforms import get_noise_transform,noise_from_cfg
from data_hub.reproduce import RandomOnce,get_random_state,enumerate_indices

# -- local imports --
from .paths import IMAGE_PATH_PAIRED as IMAGE_PATH
from .paths import IMAGE_SETS_PAIRED as IMAGE_SETS
from .paired_reader import read_files,read_video

class SubmilliluxPaired():

    def __init__(self,iroot,sroot,split,nsamples=0,nframes=0,isize=None):

        # -- set init params --
        self.iroot = iroot
        self.sroot = sroot
        self.split = split
        self.nframes = nframes
        self.isize = isize
        self.rand_crop = None if isize is None else RandomCrop(isize)

        # -- load paths --
        self.paths = read_files(iroot,sroot,split,nframes)
        self.groups = sorted(list(self.paths['images'].keys()))

        # -- limit num of samples --
        self.indices = enumerate_indices(len(self.paths['images']),nsamples)
        self.nsamples = len(self.indices)

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
        noisy_files,gt_files = self.paths['images'][group]
        noisy = read_video(noisy_files)
        clean = read_video(gt_files)

        # -- limit frame --
        if not(self.rand_crop is None):
            indices = self.rand_crop.get_params(noisy,self.isize)
            i, j, h, w = indices
            noisy = tvF.crop(noisy, i, j, h, w)
            clean = tvF.crop(clean, i, j, h, w)

        # -- manage flow and output --
        index_th = th.IntTensor([image_index])

        return {'noisy':noisy,'clean':clean,'index':index_th,
                'rng_state':rng_state}

#
# Loading the datasets in a project
#

def get_submillilux_dataset(cfg):
    return load(cfg)


def load_paired(cfg):
    return load(cfg)

def load(cfg):

    #
    # -- extract --
    #

    # -- set-up --
    modes = ['tr','val','te']

    # -- frames --
    def_nframes = optional(cfg,"nframes",0)
    nframes = edict()
    for mode in modes:
        nframes[mode] = optional(cfg,"%s_nframes"%mode,def_nframes)

    # -- frame sizes --
    def_isize = optional(cfg,"isize",None)
    isizes = edict()
    for mode in modes:
        isizes[mode] = get_isize(optional(cfg,"%s_isize"%mode,def_isize))

    # -- samples --
    def_nsamples = optional(cfg,"nsamples",-1)
    nsamples = edict()
    for mode in modes:
        nsamples[mode] = optional(cfg,"%s_nsamples"%mode,def_nsamples)

    # -- setup paths --
    iroot = IMAGE_PATH
    sroot = IMAGE_SETS

    # -- create objcs --
    data = edict()
    data.tr = SubmilliluxPaired(iroot,sroot,"train",
                          nsamples.tr,nframes.tr,isizes.tr)
    data.val = SubmilliluxPaired(iroot,sroot,"val",
                           nsamples.val,nframes.val,isizes.val)
    data.te = SubmilliluxPaired(iroot,sroot,"test",
                          nsamples.te,nframes.te,isizes.te)

    # -- create loader --
    batch_size = optional(cfg,'batch_size',1)
    loader = get_loaders(cfg,data,batch_size)

    return data,loader


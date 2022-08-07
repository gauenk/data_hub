"""
URBAN100 dataset

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
from data_hub.opt_parsing import parse_cfg
from data_hub.common import get_loaders,optional,get_isize
from data_hub.transforms import get_noise_transform,noise_from_cfg
from data_hub.reproduce import RandomOnce,get_random_state,enumerate_indices

# -- local imports --
from .paths import IMAGE_PATH,IMAGE_SETS
from .reader import read_files,read_video

class URBAN100():

    def __init__(self,iroot,sroot,split,noise_info,use_bw=False,nsamples=0,isize=None):

        # -- set init params --
        self.iroot = iroot
        self.sroot = sroot
        self.split = split
        self.isize = isize
        self.use_bw = use_bw
        self.rand_crop = None if isize is None else RandomCrop(isize)

        # -- create transforms --
        self.noise_trans = get_noise_transform(noise_info,noise_only=True)

        # -- load paths --
        self.paths = read_files(iroot,sroot,split)
        self.groups = sorted(list(self.paths['images'].keys()))

        # -- limit num of samples --
        self.indices = enumerate_indices(len(self.paths['images']),nsamples)
        self.nsamples = len(self.indices)

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
        vid_files = self.paths['images'][group]
        clean = read_video(vid_files,self.use_bw)
        clean = th.from_numpy(clean)

        # -- limit frame --
        if not(self.rand_crop is None):
            clean = self.rand_crop(clean)

        # -- get noise --
        # with self.fixRandNoise_1.set_state(index):
        noisy = self.noise_trans(clean)

        # -- elems to tensors --
        index_th = th.IntTensor([image_index])

        return {'noisy':noisy,'clean':clean,'index':index_th,
                'rng_state':rng_state}

#
# Loading the datasets in a project
#

def get_urban100_dataset(cfg):
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
              "bw":False,
              "rand_order":False}
    p = parse_cfg(cfg,modes,fields)

    # -- setup paths --
    iroot = IMAGE_PATH
    sroot = IMAGE_SETS

    # -- create objcs --
    data = edict()
    data.tr = URBAN100(iroot,sroot,"train",noise_info,
                       p.bw.tr,p.nsamples.tr,p.isize.tr)
    data.val = URBAN100(iroot,sroot,"val",noise_info,
                        p.bw.val,p.nsamples.val,p.isize.val)
    data.te = URBAN100(iroot,sroot,"test",noise_info,
                       p.bw.te,p.nsamples.te,p.isize.te)

    # -- create loader --
    loader = get_loaders(cfg,data,p.batch_size)

    return data,loader


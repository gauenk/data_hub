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
import torchvision.transforms.functional as tvF
from torchvision.transforms.functional import center_crop

# -- project imports --
from data_hub.common import get_loaders,optional,get_isize
# from data_hub.transforms import get_noise_transform,noise_from_cfg
from data_hub.reproduce import RandomOnce,get_random_state,enumerate_indices
from data_hub.cropping import crop_vid
from data_hub.opt_parsing import parse_cfg

# -- local imports --
from .paths import IMAGE_PATH_REAL as IMAGE_PATH
from .paths import IMAGE_SETS_REAL as IMAGE_SETS
from .reader import read_files,read_mats

class SubmilliluxReal():

    def __init__(self,iroot,sroot,split,noise_info,
                 nsamples=0,nframes=0,fskip=1,isize=None,
                 cropmode="center"):

        # -- set init params --
        self.iroot = iroot
        self.sroot = sroot
        self.split = split
        self.nframes = nframes
        self.fskip = fskip
        self.isize = isize
        self.crop = None if isize is None else isize

        # -- create transforms --
        self.noise_trans = None#get_noise_transform(noise_info,noise_only=True)

        # -- manage cropping --
        isize_is_none = isize is None or isize == "none"
        self.crop = isize
        self.cropmode = cropmode if not(isize_is_none) else "none"
        self.region_temp = None
        if not(isize_is_none):
            self.region_temp = "%d_%d_%d" % (nframes,isize[0],isize[1])

        # -- load paths --
        self.paths = read_files(iroot,sroot,split,nframes,fskip,"mat")
        self.groups = list(self.paths['images'].keys())

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
        noisy = read_mats(vid_files)
        noisy = th.from_numpy(noisy)

        # -- meta info --
        frame_nums = th.IntTensor(self.paths['fnums'][group])

        # -- limit frame --
        if not(self.crop is None):
            noisy = center_crop(noisy,self.crop)

        # -- manage flow and output --
        index_th = th.IntTensor([image_index])

        # -- cropping --
        region = th.IntTensor([])
        use_region = "region" in self.cropmode or "coords" in self.cropmode
        if use_region:
            region = crop_vid(noisy,self.cropmode,self.isize,self.region_temp)
        else:
            noisy = crop_vid(noisy,self.cropmode,self.isize,self.region_temp)
        clean = th.zeros_like(noisy)

        return {'noisy':noisy,'index':index_th,"clean":clean,
                'fnums':frame_nums,'region':region,
                'rng_state':rng_state}

#
# Loading the datasets in a project
#

def get_submillilux_dataset(cfg):
    return load(cfg)

def load_real(cfg):
    return load(cfg)

def load(cfg):

    #
    # -- extract --
    #

    # -- noise and dyanmics --
    # noise_info = noise_from_cfg(cfg)
    noise_info = None

    # -- set-up --
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
              "cropmode":"region"}
    p = parse_cfg(cfg,modes,fields)

    # -- setup paths --
    iroot = IMAGE_PATH
    sroot = IMAGE_SETS

    # -- create objcs --
    data = edict()
    data.tr = SubmilliluxReal(iroot,sroot,"train",noise_info,
                              p.nsamples.tr,p.nframes.tr,p.fskip.tr,p.isize.tr,
                              p.cropmode.tr)
    data.val = SubmilliluxReal(iroot,sroot,"val",noise_info,
                               p.nsamples.val,p.nframes.val,p.fskip.val,p.isize.val,
                               p.cropmode.val)
    data.te = SubmilliluxReal(iroot,sroot,"test",noise_info,
                              p.nsamples.te,p.nframes.te,p.fskip.te,p.isize.te,
                              p.cropmode.te)

    # -- create loader --
    loader = get_loaders(cfg,data,p.batch_size)

    return data,loader


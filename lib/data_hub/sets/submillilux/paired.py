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
from data_hub.cropping import crop_vid
from data_hub.opt_parsing import parse_cfg

# -- local imports --
from .paths import IMAGE_PATH_PAIRED as IMAGE_PATH
from .paths import IMAGE_SETS_PAIRED as IMAGE_SETS
from .paired_reader import read_files,read_video

class SubmilliluxPaired():

    def __init__(self,iroot,sroot,split,nsamples=0,nframes=0,
                 isize=None,cropmode="center",bw=False):

        # -- set init params --
        self.iroot = iroot
        self.sroot = sroot
        self.split = split
        self.nframes = nframes
        self.isize = isize
        self.bw = bw

        # -- manage cropping --
        isize_is_none = isize is None or isize == "none"
        self.crop = isize
        self.cropmode = cropmode if not(isize_is_none) else "none"
        self.region_temp = None
        if not(isize_is_none):
            self.region_temp = "%d_%d_%d" % (nframes,isize[0],isize[1])

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
        noisy = read_video(noisy_files,self.bw)
        clean = read_video(gt_files,self.bw)
        print(noisy_files)
        print(gt_files)

        # -- meta info --
        frame_nums = th.IntTensor(self.paths['fnums'][group])

        # -- cropping --
        region = th.IntTensor([])
        use_region = "region" in self.cropmode or "coords" in self.cropmode
        if use_region:
            region = crop_vid(clean,self.cropmode,self.isize,self.region_temp)
        else:
            clean = crop_vid(clean,self.cropmode,self.isize,self.region_temp)

        # # -- limit frame --
        # if not(self.rand_crop is None):
        #     indices = self.rand_crop.get_params(noisy,self.isize)
        #     i, j, h, w = indices
        #     noisy = tvF.crop(noisy, i, j, h, w)
        #     clean = tvF.crop(clean, i, j, h, w)

        # -- manage flow and output --
        index_th = th.IntTensor([image_index])

        return {'noisy':noisy,'clean':clean,'index':index_th,
                'rng_state':rng_state,'fnums':frame_nums,'region':region}

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
              "cropmode":"center"}
    p = parse_cfg(cfg,modes,fields)

    # -- setup paths --
    iroot = IMAGE_PATH
    sroot = IMAGE_SETS

    # -- create objcs --
    data = edict()
    data.tr = SubmilliluxPaired(iroot,sroot,"train",
                                p.nsamples.tr,p.nframes.tr,
                                p.isize.tr,p.cropmode.tr,p.bw.tr)
    data.val = SubmilliluxPaired(iroot,sroot,"val",
                                 p.nsamples.val,p.nframes.val,
                                 p.isize.val,p.cropmode.val,p.bw.val)
    data.te = SubmilliluxPaired(iroot,sroot,"test",
                                p.nsamples.te,p.nframes.te,
                                p.isize.te,p.cropmode.val,p.bw.te)

    # -- create loader --
    loader = get_loaders(cfg,data,p.batch_size)

    return data,loader


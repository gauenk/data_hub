"""
SIRD dataset

Single Image Real Deraining

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
from .paths import IMAGE_PATH,IMAGE_SETS
from .reader import read_files,read_video

class SIRD():

    def __init__(self,iroot,sroot,split,
                 nsamples=0,nframes=0,fstride=1,isize=None,
                 bw=False,cropmode="coords",rand_order=False,
                 index_skip=1):

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

        # -- load paths --
        self.paths = read_files(iroot,sroot,split,nframes,fstride)
        self.groups = sorted(list(self.paths['images'].keys()))

        # -- limit num of samples --
        self.indices = enumerate_indices(len(self.paths['images']),nsamples,
                                         rand_order,index_skip)
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
        vid_files = self.paths['images'][group]
        xy = self.paths['xy'][group]
        noisy,clean = read_video(vid_files,xy,self.bw)

        # -- meta info --
        frame_nums = th.IntTensor(self.paths['fnums'][group])

        # -- cropping --
        region = th.IntTensor([])
        use_region = "region" in self.cropmode or "coords" in self.cropmode
        if use_region:
            region = crop_vid(clean,self.cropmode,self.isize,self.region_temp)
        else:
            noisy,clean = crop_vid([noisy,clean],self.cropmode,
                                   self.isize,self.region_temp)

        # -- manage flow and output --
        index_th = th.IntTensor([image_index])

        return {'noisy':noisy,'clean':clean,'index':index_th,
                'fnums':frame_nums,'region':region,'rng_state':rng_state}

#
# Loading the datasets in a project
#

def get_sird_dataset(cfg):
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
              "cropmode":"region"}
    p = parse_cfg(cfg,modes,fields)

    # -- setup paths --
    iroot = IMAGE_PATH
    sroot = IMAGE_SETS

    # -- create objcs --
    data = edict()
    data.tr = SIRD(iroot,sroot,"train",p.nsamples.tr,p.nframes.tr,
                   p.fstride.tr,p.isize.tr,p.bw.tr,p.cropmode.tr,
                   p.rand_order.tr,p.index_skip.tr)
    data.val = SIRD(iroot,sroot,"val",p.nsamples.val,p.nframes.val,
                    p.fstride.val,p.isize.val,p.bw.val,p.cropmode.tr,
                    p.rand_order.val,p.index_skip.val)
    data.te = SIRD(iroot,sroot,"test",p.nsamples.val,p.nframes.val,
                   p.fstride.val,p.isize.val,p.bw.val,p.cropmode.tr,
                   p.rand_order.val,p.index_skip.val)
    loader = get_loaders(cfg,data,p.batch_size)

    return data,loader


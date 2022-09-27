"""
GoPro dataset

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
# from data_hub.transforms import get_noise_transform,noise_from_cfg
from data_hub.reproduce import RandomOnce,get_random_state,enumerate_indices
from data_hub.cropping import crop_vid
from data_hub.opt_parsing import parse_cfg

# -- local imports --
from .paths import BASE
from .reader_cropped import read_names,read_data

# -- augmentations init --
import random
from data_hub.augmentations import Augment_RGB_Flips
augment = Augment_RGB_Flips()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]



class GoProCropped():

    def __init__(self,iroot,split,
                 nsamples=0,nframes=0,fstride=1,isize=None,
                 bw=False,cropmode=None,rand_order=False,
                 index_skip=1):

        # -- set init params --
        self.iroot = iroot
        self.split = split
        self.nframes = nframes
        self.isize = isize
        self.rand_order = rand_order
        self.index_skip = index_skip
        assert self.nframes <= 10,"Must be <= 10 for this dataset."

        # -- manage cropping --
        isize_is_none = isize is None or isize == "none"
        self.crop = isize
        self.cropmode = cropmode if not(isize_is_none) else "none"
        self.region_temp = None
        if not(isize_is_none):
            self.region_temp = "%d_%d_%d" % (nframes,isize[0],isize[1])

        # -- load paths --
        self.names = read_names(iroot,self.nframes,ext="jpg")
        self.groups = sorted(self.names)

        # -- limit num of samples --
        self.indices = enumerate_indices(len(self.names),
                                         nsamples,rand_order,index_skip)
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
        vid_name = self.names[image_index]
        blur,sharp,frame_nums = read_data(vid_name,self.iroot,self.nframes)

        # -- cropping --
        region = th.IntTensor([])
        use_region = "region" in self.cropmode or "coords" in self.cropmode
        if use_region:
            region = crop_vid(sharp,self.cropmode,self.isize,self.region_temp)
        else:
            vids = crop_vid([sharp,blur],self.cropmode,self.isize,self.region_temp)
            sharp,blur = vids

        # -- augmentations --
        apply_trans = transforms_aug[random.getrandbits(3)]
        sharp = getattr(augment, apply_trans)(sharp)
        blur = getattr(augment, apply_trans)(blur)

        # -- manage flow and output --
        index_th = th.IntTensor([image_index])

        return {'blur':blur,'sharp':sharp,
                'index':index_th,'fnums':frame_nums,'region':region,
                'rng_state':rng_state}

#
# Loading the datasets in a project
#

def get_gopro_dataset(cfg):
    return load(cfg)


def load(cfg):

    #
    # -- extract --
    #

    # -- field names and defaults --
    modes = ['tr','val','te']
    fields = {"bw":False,
              "nframes":10,
              "fstride":1,
              "isize":None,
              "nsamples":-1,
              "fskip":1,
              "index_skip":1,
              "batch_size":1,
              "rand_order":True,
              "cropmode":None}
    p = parse_cfg(cfg,modes,fields)

    # -- setup paths --
    iroot = BASE

    # -- create objcs --
    data = edict()
    data.tr = GoProCropped(iroot,"train",p.nsamples.tr,p.nframes.tr,
                           p.fstride.tr,p.isize.tr,p.bw.tr,
                           p.cropmode.tr,p.rand_order.tr,p.index_skip.tr)
    data.val = GoProCropped(iroot,"train",p.nsamples.val,p.nframes.val,
                            p.fstride.val,p.isize.val,p.bw.val,
                            p.cropmode.tr,p.rand_order.val,p.index_skip.val)
    data.te = GoProCropped(iroot,"train",p.nsamples.val,p.nframes.val,
                           p.fstride.val,p.isize.val,p.bw.val,
                           p.cropmode.tr,p.rand_order.val,p.index_skip.val)
    loader = get_loaders(cfg,data,p.batch_size)

    return data,loader


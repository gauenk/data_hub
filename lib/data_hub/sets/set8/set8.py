"""
Set8 dataset

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

# -- local imports --
from .paths import IMAGE_PATH,IMAGE_SETS
from .reader import read_files,read_video

class Set8():

    def __init__(self,iroot,sroot,split,noise_info,
                 nsamples=0,nframes=0,fskip=1,isize=None):

        # -- set init params --
        self.iroot = iroot
        self.sroot = sroot
        self.split = split
        self.nframes = nframes
        self.isize = isize

        # -- manage cropping --
        self.rand_crop = None if isize is None else RandomCrop(isize)
        self.crop = isize

        # -- create transforms --
        self.noise_trans = get_noise_transform(noise_info,noise_only=True)

        # -- load paths --
        self.paths = read_files(iroot,sroot,split,nframes,fskip)
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
        clean = read_video(vid_files)
        clean = th.from_numpy(clean)

        # -- meta info --
        frame_nums = self.paths['fnums'][group]

        # -- limit frame --
        if not(self.rand_crop is None):
            clean = center_crop(clean,self.isize)
            # clean = self.rand_crop(clean)

        # -- get noise --
        # with self.fixRandNoise_1.set_state(index):
        noisy = self.noise_trans(clean)

        # -- manage flow and output --
        index_th = th.IntTensor([image_index])

        return {'noisy':noisy,'clean':clean,'index':index_th,
                'fnums':frame_nums,'rng_state':rng_state}

#
# Loading the datasets in a project
#

def get_set8_dataset(cfg):
    return load(cfg)


def load(cfg):

    #
    # -- extract --
    #

    # -- noise and dyanmics --
    noise_info = noise_from_cfg(cfg)

    # -- set-up --
    modes = ['tr','val','te']

    # -- frames --
    def_nframes = optional(cfg,"nframes",0)
    nframes = edict()
    for mode in modes:
        nframes[mode] = optional(cfg,"nframes_%s"%mode,def_nframes)

    # -- fskip [amount of overlap for subbursts] --
    def_fskip = optional(cfg,"fskip",1)
    fskip = edict()
    for mode in modes:
        fskip[mode] = optional(cfg,"%s_fskip"%mode,def_fskip)

    # -- frame sizes --
    def_isize = optional(cfg,"isize",None)
    if def_isize == "-1_-1": def_size = None
    isizes = edict()
    for mode in modes:
        isizes[mode] = get_isize(optional(cfg,"isize_%s"%mode,def_isize))

    # -- samples --
    def_nsamples = optional(cfg,"nsamples",-1)
    nsamples = edict()
    for mode in modes:
        nsamples[mode] = optional(cfg,"nsamples_%s"%mode,def_nsamples)

    # -- setup paths --
    iroot = IMAGE_PATH
    sroot = IMAGE_SETS

    # -- create objcs --
    data = edict()
    data.tr = Set8(iroot,sroot,"train",noise_info,nsamples.tr,
                   nframes.tr,fskip.tr,isizes.tr)
    data.val = Set8(iroot,sroot,"val",noise_info,nsamples.val,
                    nframes.val,fskip.val,isizes.val)
    data.te = Set8(iroot,sroot,"test",noise_info,nsamples.te,
                   nframes.te,fskip.te,isizes.te)

    # -- create loader --
    batch_size = optional(cfg,'batch_size',1)
    loader = get_loaders(cfg,data,batch_size)

    return data,loader


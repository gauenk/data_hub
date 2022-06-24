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
import scipy.io
from data_hub.common import get_loaders,optional,get_isize
# from data_hub.transforms import get_noise_transform,noise_from_cfg
from data_hub.reproduce import RandomOnce,get_random_state,enumerate_indices

# -- local imports --
from .paths import IMAGE_ROOT
# from .reader import read_files,read_mats

class SIDDRgbVal():

    def __init__(self,iroot,nsamples=0,nframes=0,fskip=1,isize=None):

        # -- set init params --
        self.iroot = iroot
        self.nsamples = nsamples
        self.nframes = nframes
        self.fskip = fskip
        self.isize = isize
        self.crop = None if ((isize is None) or (isize == "none")) else isize

        # -- load paths --
        self.noisy_fn = iroot / "ValidationNoisyBlocksSrgb.mat"
        self.clean_fn = iroot / "ValidationGtBlocksSrgb.mat"
        self.noisy = scipy.io.loadmat(self.noisy_fn)['ValidationNoisyBlocksSrgb']
        self.clean = scipy.io.loadmat(self.clean_fn)['ValidationGtBlocksSrgb']
        self.groups = np.array(["%02d" % x for x in range(len(self.noisy))])

    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            data sample.
        """

        # -- random once --
        rng_state = None#get_random_state()

        # -- get image index --
        image_index = index

        # -- get image chunks --
        noisy = self.noisy[index]
        clean = self.clean[index]

        # -- reshape --
        noisy = rearrange(noisy,'g h w c -> g c h w')
        clean = rearrange(clean,'g h w c -> g c h w')

        # -- type --
        noisy = noisy.astype(np.float32)
        clean = clean.astype(np.float32)

        # -- to torch --
        noisy = th.from_numpy(noisy).contiguous()
        clean = th.from_numpy(clean).contiguous()

        # -- meta info --
        frame_nums = np.arange(noisy.shape[0])

        # -- limit frame --
        if not(self.crop is None):
            noisy = center_crop(noisy,self.crop)
            clean = center_crop(clean,self.crop)

        # -- manage flow and output --
        index_th = th.IntTensor([image_index])

        return {'noisy':noisy,'clean':clean,
                'index':index_th,'fnums':frame_nums,
                'rng_state':rng_state}

#
# Loading the datasets in a project
#

def get_sidd_dataset(cfg):
    return load(cfg)

def load_val(cfg):
    return load(cfg)

def load(cfg):

    #
    # -- extract --
    #

    # -- set-up --
    modes = ['val']

    # -- frames --
    def_nframes = optional(cfg,"nframes",0)
    nframes = edict()
    for mode in modes:
        nframes[mode] = optional(cfg,"%s_nframes"%mode,def_nframes)

    # -- fskip [amount of overlap for subbursts] --
    def_fskip = optional(cfg,"fskip",1)
    fskip = edict()
    for mode in modes:
        fskip[mode] = optional(cfg,"%s_fskip"%mode,def_fskip)

    # -- frame sizes --
    def_isize = optional(cfg,"isize","none")
    isizes = edict()
    for mode in modes:
        isizes[mode] = get_isize(optional(cfg,"%s_isize"%mode,def_isize))

    # -- samples --
    def_nsamples = optional(cfg,"nsamples",-1)
    nsamples = edict()
    for mode in modes:
        nsamples[mode] = optional(cfg,"%s_nsamples"%mode,def_nsamples)

    # -- create objcs --
    data = edict()
    data.val = SIDDRgbVal(IMAGE_ROOT,nsamples.val,nframes.val,
                          fskip.val,isizes.val)

    # -- create loader --
    batch_size = optional(cfg,'batch_size',1)
    loader = get_loaders(cfg,data,batch_size)

    return data,loader


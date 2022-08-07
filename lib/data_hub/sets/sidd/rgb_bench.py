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
from data_hub.opt_parsing import parse_cfg

# -- local imports --
from .paths import IMAGE_ROOT
# from .reader import read_files,read_mats

class SIDDRgbBench():

    def __init__(self,iroot,nsamples=0,nframes=0,fskip=1,isize=None):

        # -- set init params --
        self.iroot = iroot
        self.nsamples = nsamples
        self.nframes = nframes
        self.fskip = fskip
        self.isize = isize
        self.crop = None if ((isize is None) or (isize == "none")) else isize

        # -- load paths --
        self.noisy_fn = iroot / "BenchmarkNoisyBlocksSrgb.mat"
        self.noisy = scipy.io.loadmat(self.noisy_fn)['BenchmarkNoisyBlocksSrgb']
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
        noisy = rearrange(noisy,'g h w c -> g c h w')

        # -- type --
        noisy = noisy.astype(np.float32)

        # -- to torch --
        noisy = th.from_numpy(noisy).contiguous()

        # -- meta info --
        frame_nums = np.arange(noisy.shape[0])

        # -- limit frame --
        if not(self.crop is None):
            noisy = center_crop(noisy,self.crop)

        # -- manage flow and output --
        index_th = th.IntTensor([image_index])

        return {'noisy':noisy,'clean':noisy,
                'index':index_th,'fnums':frame_nums,
                'rng_state':rng_state}

#
# Loading the datasets in a project
#

def get_sidd_dataset(cfg):
    return load(cfg)

def load_bench(cfg):
    return load(cfg)

def load(cfg):

    #
    # -- extract --
    #

    # -- set-up --
    modes = ['val']

    # -- field names and defaults --
    fields = {"batch_size":1,
              "nsamples":-1,
              "isize":None,
              "fskip":1,
              "fstride":1,
              "nframes":0,
              "rand_order":False,
              "cropmode":"region"}
    p = parse_cfg(cfg,modes,fields)

    # -- create objcs --
    data = edict()
    data.val = SIDDRgbBench(IMAGE_ROOT,p.nsamples.val,p.nframes.val,
                            p.fskip.val,p.isize.val)

    # -- create loader --
    loader = get_loaders(cfg,data,p.batch_size)

    return data,loader


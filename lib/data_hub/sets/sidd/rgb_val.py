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
from data_hub.cropping import crop_vid
from data_hub.opt_parsing import parse_cfg

# -- local imports --
from .paths import IMAGE_ROOT
# from .reader import read_files,read_mats

class SIDDRgbVal():

    def __init__(self,iroot,nsamples=0,nframes=0,fskip=1,isize=None,
                 cropmode="coords",rand_order=False,index_skip=1):

        # -- set init params --
        self.iroot = iroot
        self.nsamples = nsamples
        self.nframes = nframes
        self.fskip = fskip
        self.isize = isize

        # -- manage cropping --
        isize_is_none = isize is None or isize == "none"
        self.crop = isize
        self.cropmode = cropmode if not(isize_is_none) else "none"
        self.region_temp = None
        if not(isize_is_none):
            self.region_temp = "%d_%d_%d" % (nframes,isize[0],isize[1])

        # -- load paths --
        self.noisy_fn = iroot / "ValidationNoisyBlocksSrgb.mat"
        self.clean_fn = iroot / "ValidationGtBlocksSrgb.mat"
        self.noisy = scipy.io.loadmat(self.noisy_fn)['ValidationNoisyBlocksSrgb']
        self.clean = scipy.io.loadmat(self.clean_fn)['ValidationGtBlocksSrgb']
        self.groups = np.array(["%02d" % x for x in range(len(self.noisy))])
        self.blocks = None # an api requirement, see "...full.py" for comparison

        # -- limit num of samples --
        self.indices = enumerate_indices(len(self.noisy),nsamples,
                                         rand_order,index_skip)
        self.nsamples = len(self.indices)


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
        frame_nums = th.IntTensor(np.arange(noisy.shape[0]))

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

        return {'noisy':noisy,'clean':clean,"region":region,
                'index':index_th,'fnums':frame_nums,'rng_state':rng_state}

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

    # -- field names and defaults --
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

    # -- create objcs --
    data = edict()
    data.val = SIDDRgbVal(IMAGE_ROOT,p.nsamples.val,p.nframes.val,p.fskip.val,
                          p.isize.val,p.cropmode.val,p.rand_order.val,p.index_skip.val)

    # -- create loader --
    loader = get_loaders(cfg,data,p.batch_size)

    return data,loader


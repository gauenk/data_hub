"""
SIDD dataset

"""

# -- python imports --
import pdb
try:
    import hdf5storage
except:
    pass
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

# -- image --
from PIL import Image

# -- local imports --
from .paths import IMAGE_ROOT
from .reader import read_files

class SIDDBenchFull():

    def __init__(self,iroot,mode,nsamples=0,nframes=0,fskip=1,isize=None):

        # -- set init params --
        self.iroot = iroot
        self.mode = mode
        self.nsamples = nsamples
        self.nframes = nframes
        self.fskip = fskip
        self.isize = isize
        self.crop = None if ((isize is None) or (isize == "none")) else isize

        # -- load paths --
        iroot = iroot / "SIDD_Benchmark_Data"
        self.block_fn = iroot / "BenchmarkBlocks32.mat"
        self.files,self.groups = read_files(iroot,mode)
        self.blocks = scipy.io.loadmat(self.block_fn)['BenchmarkBlocks32']

    def __len__(self):
        return self.nsamples


    def read_file(self,index):
        if self.mode == "rgb":
            noisy = Image.open(self.files[index])
            noisy = noisy.convert("RGB")
        elif self.mode == "raw":
            fn = str(self.files[index])
            noisy = hdf5storage.loadmat(fn)['x']
            noisy = noisy[...,None]
        noisy = np.array(noisy)[None,:]
        return noisy

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
        noisy = self.read_file(index)
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

    # -- batch size --
    def_batch_size = optional(cfg,'batch_size',1)
    batch_size = edict()
    for mode in modes:
        batch_size[mode] = optional(cfg,'batch_size_%s'%mode,def_batch_size)

    # -- create objcs --
    data = edict()
    bench_mode = optional(cfg,"sidd_bench_mode","rgb") # or "raw"
    data.val = SIDDBenchFull(IMAGE_ROOT,bench_mode,nsamples.val,nframes.val,
                             fskip.val,isizes.val)

    # -- create loader --
    loader = get_loaders(cfg,data,batch_size)

    return data,loader


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
from data_hub.transforms import get_noise_transform,noise_from_cfg
from data_hub.reproduce import RandomOnce,get_random_state,enumerate_indices
from data_hub.cropping import apply_sobel_filter,sample_sobel_region,sample_rand_region

# -- local imports --
from .paths import BASE
from .reader import read_files,read_data

class GoPro():

    def __init__(self,iroot,split,noise_info,
                 nsamples=0,nframes=0,fstride=1,isize=None,
                 bw=False,cropmode="coords",rand_order=False,
                 index_skip=1):

        # -- set init params --
        self.iroot = iroot
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
        self.rand_crop,self.region_temp = None,None
        if not(isize_is_none):
            self.rand_crop = RandomCrop(isize)
            self.region_temp = "%d_%d_%d" % (nframes,isize[0],isize[1])

        # -- create transforms --
        self.noise_trans = get_noise_transform(noise_info,noise_only=True)

        # -- load paths --
        self.paths = read_files(iroot,split,nframes,fstride,ext="jpg")
        self.groups = sorted(list(self.paths['images'].keys()))

        # -- limit num of samples --
        self.indices = enumerate_indices(len(self.paths['images']),nsamples,rand_order,index_skip)
        self.nsamples = len(self.indices)

        # -- repro --
        self.noise_once = optional(noise_info,"sim_once",False)
        # self.fixRandNoise_1 = RandomOnce(self.noise_once,self.nsamples)

    def __len__(self):
        return self.nsamples

    def _crop_vid(self,vid):
        if self.cropmode is None or self.cropmode == "none": return vid
        if self.cropmode == "rand":
            vid = self.rand_crop(vid,self.isize)
            rtn = vid
        elif self.cropmode in ["coords","coords_sobel","region","region_sobel"]:
            sobel_vid = apply_sobel_filter(vid)
            region = sample_sobel_region(sobel_vid,self.region_temp)
            region = th.IntTensor(region)
            rtn = region
        elif self.cropmode in ["coords_rand","region_rand"]:
            sobel_vid = apply_sobel_filter(vid)
            region = sample_rand_region(sobel_vid,self.region_temp)
            region = th.IntTensor(region)
            rtn = region
        elif self.cropmode in ["center_crop","center"]:
            vid_cc = center_crop(vid,self.isize)
            rtn = vid_cc
        else:
            raise NotImplementedError(f"Uknown crop mode [{self.cropmode}]")
        return rtn

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
        data = read_data(vid_files,self.bw)

        # -- unpack --
        sharp = data.sharp
        blur = data.blur
        blur_gamma = data.blur_gamma

        # -- meta info --
        frame_nums = th.IntTensor(self.paths['fnums'][group])

        # -- cropping --
        region = th.IntTensor([])
        use_region = "region" in self.cropmode or "coords" in self.cropmode
        if use_region:
            region = self._crop_vid(blur)
        else:
            blur = self._crop_vid(blur)
            sharp = self._crop_vid(sharp)
            blur_gamma = self._crop_vid(blur_gamma)

        # -- manage flow and output --
        index_th = th.IntTensor([image_index])

        return {'blur':blur,'sharp':sharp,'blur_gamma':blur_gamma,
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

    # -- noise and dyanmics --
    noise_info = noise_from_cfg(cfg)

    # -- set-up --
    modes = ['tr','val','te']

    # -- bw --
    def_bw = optional(cfg,"bw",False)
    bw = edict()
    for mode in modes:
        bw[mode] = optional(cfg,"bw_%s"%mode,def_bw)

    # -- frames --
    def_nframes = optional(cfg,"nframes",0)
    nframes = edict()
    for mode in modes:
        nframes[mode] = optional(cfg,"nframes_%s"%mode,def_nframes)

    # -- fstride [amount of overlap for subbursts] --
    def_fstride = optional(cfg,"fstride",1)
    fstride = edict()
    for mode in modes:
        fstride[mode] = optional(cfg,"%s_fstride"%mode,def_fstride)

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

    # -- crop mode --
    def_cropmode = optional(cfg,"cropmode","region")
    cropmode = edict()
    for mode in modes:
        cropmode[mode] = optional(cfg,"cropmode_%s"%mode,def_cropmode)

    # -- random order --
    def_rand_order = optional(cfg,'rand_order',False)
    rand_order = edict()
    for mode in modes:
        rand_order[mode] = optional(cfg,"rand_order_%s"%mode,def_rand_order)

    # -- random order --
    def_index_skip = optional(cfg,'index_skip',1)
    index_skip = edict()
    for mode in modes:
        index_skip[mode] = optional(cfg,"index_skip_%s"%mode,def_index_skip)

    # -- batch size --
    def_batch_size = optional(cfg,'batch_size',1)
    batch_size = edict()
    for mode in modes:
        batch_size[mode] = optional(cfg,'batch_size_%s'%mode,def_batch_size)

    # -- setup paths --
    iroot = BASE

    # -- create objcs --
    data = edict()
    data.tr = GoPro(iroot,"train",noise_info,nsamples.tr,
                    nframes.tr,fstride.tr,isizes.tr,bw.tr,cropmode.tr,
                    rand_order.tr,index_skip.tr)
    data.te = GoPro(iroot,"test",noise_info,nsamples.val,
                    nframes.val,fstride.val,isizes.val,bw.val,cropmode.tr,
                    rand_order.val,index_skip.val)
    loader = get_loaders(cfg,data,batch_size)

    return data,loader


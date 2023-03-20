"""
DAVIS dataset

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

# -- optical flow --
import torch as th
from .paths import FLOW_BASE # why not other paths? I think we can do it when time

# -- local imports --
from .paths import IMAGE_PATH,IMAGE_SETS
from .reader import read_files,read_video,read_flows


class DAVIS():

    def __init__(self,iroot,sroot,split,noise_info,params):
                 # nsamples=0,nframes=0,fstride=1,isize=None,
                 # bw=False,cropmode="coords",rand_order=False,
                 # index_skip=1):

        # -- set init params --
        self.iroot = iroot
        self.sroot = sroot
        self.split = split
        self.nframes = params.nframes
        self.isize = params.isize
        self.bw = params.bw
        self.rand_order = params.rand_order
        self.index_skip = params.index_skip
        self.read_flows = params.read_flows
        self.seed = params.seed
        self.noise_info = noise_info

        # -- manage cropping --
        isize = params.isize
        isize_is_none = isize is None or isize == "none"
        self.crop = isize
        self.cropmode = params.cropmode if not(isize_is_none) else "none"
        self.region_temp = None
        if not(isize_is_none):
            self.region_temp = "%d_%d_%d" % (params.nframes,isize[0],isize[1])

        # -- create transforms --
        self.noise_trans = get_noise_transform(noise_info,noise_only=True)

        # -- load paths --
        self.paths = read_files(iroot,sroot,split,params.nframes,
                                params.fstride,ext="jpg")
        self.groups = sorted(list(self.paths['images'].keys()))

        # -- limit num of samples --
        self.indices = enumerate_indices(len(self.paths['images']),params.nsamples,
                                         params.rand_order,params.index_skip)
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
        clean = read_video(vid_files,self.bw)
        clean = th.from_numpy(clean)

        # -- meta info --
        frame_nums = th.IntTensor(self.paths['fnums'][group])

        # -- cropping --
        region = th.IntTensor([])
        use_region = "region" in self.cropmode or "coords" in self.cropmode
        if use_region:
            region = crop_vid(clean,self.cropmode,self.isize,self.region_temp)
        else:
            clean = crop_vid(clean,self.cropmode,self.isize,self.region_temp)

        # -- get noise --
        # with self.fixRandNoise_1.set_state(index):
        noisy = self.noise_trans(clean)

        # -- manage flow and output --
        index_th = th.IntTensor([image_index])

        # -- flow io --
        fflow,bflow = read_flows(self.read_flows,group,self.noise_info,self.seed)

        return {'noisy':noisy,'clean':clean,'index':index_th,
                'fnums':frame_nums,'region':region,'rng_state':rng_state,
                'fflow':fflow,'bflow':bflow}

#
# Loading the datasets in a project
#

def get_davis_dataset(cfg):
    return load(cfg)


def load(cfg):

    #
    # -- extract --
    #

    # -- noise and dyanmics --
    noise_info = noise_from_cfg(cfg)

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
              "cropmode":"center",
              "num_workers":2,
              "read_flows":False,
              "seed":123}
    p = parse_cfg(cfg,modes,fields)

    # -- setup paths --
    iroot = IMAGE_PATH
    sroot = IMAGE_SETS

    # -- create objcs --
    data = edict()
    data.tr = DAVIS(iroot,sroot,"train",noise_info,p.tr)
    data.val = DAVIS(iroot,sroot,"val",noise_info,p.val)

    # -- create loaders --
    batch_size = edict({key:val['batch_size'] for key,val in p.items()})
    loader = get_loaders(cfg,data,batch_size)

    return data,loader


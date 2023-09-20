"""
IPhoneSpring2023 dataset

"""

# -- python imports --
import pdb
import numpy as np
from pathlib import Path
from einops import rearrange,repeat
from easydict import EasyDict as edict
import copy
dcopy = copy.deepcopy

# -- pytorch imports --
import torch as th
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as tvF
from torchvision.transforms.functional import center_crop

# -- project imports --
from data_hub.common import get_loaders,optional,get_isize
from data_hub.transforms import get_noise_transform,noise_from_cfg
from data_hub.reproduce import RandomOnce,get_random_state,enumerate_indices
from data_hub.opt_parsing import parse_cfg
# from data_hub.cropping import apply_sobel_filter,sample_sobel_region,sample_rand_region,get_center_region
from data_hub.cropping import crop_vid
from data_hub.read_flow import read_flows

# -- local imports --
from .paths import IMAGE_PATH,IMAGE_SETS,FLOW_BASE
from .reader import read_files,read_video

class IPhoneSpring2023():

    def __init__(self,iroot,sroot,split,noise_info,params):

        # -- set init params --
        self.iroot = iroot
        self.sroot = sroot
        self.split = split
        self.nframes = params.nframes
        self.isize = params.isize
        self.bw = params.bw
        self.read_flows = params.read_flows
        self.seed = params.seed
        self.noise_info = noise_info

        # -- manage cropping --
        isize = params.isize
        isize_is_none = isize is None or isize == "none"
        self.crop = isize
        self.cropmode = params.cropmode if not(isize_is_none) else "none"
        self.rand_crop,self.region_temp = None,None
        if not(isize_is_none):
            self.rand_crop = RandomCrop(isize)
            self.region_temp = "%d_%d_%d" % (params.nframes,isize[0],isize[1])

        # -- create transforms --
        self.noise_trans = get_noise_transform(noise_info,noise_only=True)

        # -- load paths --
        self.paths = read_files(iroot,sroot,split,params.nframes,
                                params.fstride,params.video_seq_max)
        self.groups = sorted(list(self.paths['images'].keys()))

        # -- limit num of samples --
        self.indices = enumerate_indices(len(self.paths['images']),params.nsamples)
        self.nsamples = len(self.indices)

        # -- repro --
        self.noise_once = params.noise_once
        self.random_once = RandomOnce(self.noise_once,self.nsamples)

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
        frame_nums = self.paths['fnums'][group]
        frame_nums = th.IntTensor(frame_nums)

        # -- flow io --
        vid_name = group.split(":")[0]
        isize = list(clean.shape[-2:])
        loc = [0,len(clean),0,0]
        fflow,bflow = read_flows(FLOW_BASE,self.read_flows,vid_name,
                                 self.noise_info,self.seed,loc,isize)

        # -- cropping --
        region = th.IntTensor([])
        in_vids = [clean,fflow,bflow] if self.read_flows else [clean]
        use_region = "region" in self.cropmode or "coords" in self.cropmode
        if use_region:
            region = crop_vid(clean,self.cropmode,self.isize,self.region_temp)
        else:
            in_vids = crop_vid(in_vids,self.cropmode,self.isize,self.region_temp)
            clean = in_vids[0]
            if self.read_flows:
                fflow,bflow = in_vids[1],in_vids[2]

        # -- get noise --
        # with self.random_once.set_state(index):
        # with self.fixRandNoise_1.set_state(index):
        noisy = self.noise_trans(clean)

        # -- image index in expanded dataset [with crops] --
        index_th = th.IntTensor([image_index])

        return {'noisy':noisy,'clean':clean,'index':index_th,
                'fnums':frame_nums,'region':region,'rng_state':rng_state,
                'fflow':fflow,'bflow':bflow}

#
# Loading the datasets in a project
#

def get_iphone_s2023_dataset(cfg):
    return load(cfg)


def load(cfg):

    #
    # -- extract --
    #

    # -- noise and dyanmics --
    cfg = dcopy(cfg)
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
              "noise_once":False,
              "num_workers":2,
              "read_flows":False,
              "seed":123,
              "video_seq_max":0}
    p = parse_cfg(cfg,modes,fields)

    # -- setup paths --
    iroot = IMAGE_PATH
    sroot = IMAGE_SETS

    # -- create objcs --
    iphone_type = cfg.iphone_type # required
    data = edict()
    data.tr = IPhoneSpring2023(iroot,sroot,iphone_type,noise_info,p.tr)
    data.val = IPhoneSpring2023(iroot,sroot,"all",noise_info,p.val)
    data.te = IPhoneSpring2023(iroot,sroot,"all",noise_info,p.te)

    # -- create loader --
    batch_size = edict({key:val['batch_size'] for key,val in p.items()})
    loader = get_loaders(cfg,data,batch_size)

    return data,loader


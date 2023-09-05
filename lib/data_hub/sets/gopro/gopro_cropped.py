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
from .paths import BASE,FLOW_BASE
from .reader_cropped import read_names,read_data
from data_hub.read_flow import read_flows

# -- augmentations init --
import random
from data_hub.augmentations import get_scale_augs,get_flippy_augs
from data_hub.read_flow import read_flows
# import random
# from data_hub.augmentations import Augment_RGB_Flips
# augment = Augment_RGB_Flips()
# transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]


class GoProCropped():

    def __init__(self,iroot,split,params):

        # -- set init params --
        self.iroot = iroot
        self.split = split
        self.nframes = params.nframes
        self.isize = params.isize
        self.bw = params.bw
        self.rand_order = params.rand_order
        self.index_skip = params.index_skip
        self.read_flows = params.read_flows
        self.seed = params.seed


        # -- manage cropping --
        isize = params.isize
        cropmode = params.cropmode
        isize_is_none = isize is None or isize == "none"
        self.crop = isize
        self.cropmode = cropmode if not(isize_is_none) else "none"
        self.region_temp = None
        if not(isize_is_none):
            self.region_temp = "%d_%d_%d" % (params.nframes,isize[0],isize[1])

        # # -- set init params --
        # self.iroot = iroot
        # self.split = split
        # self.nframes = nframes
        # self.isize = isize
        # self.rand_order = rand_order
        # self.index_skip = index_skip
        # assert self.nframes <= 10,"Must be <= 10 for this dataset."

        # # -- manage cropping --
        # isize_is_none = isize is None or isize == "none"
        # self.crop = isize
        # self.cropmode = cropmode if not(isize_is_none) else "none"
        # self.region_temp = None
        # if not(isize_is_none):
        #     self.region_temp = "%d_%d_%d" % (nframes,isize[0],isize[1])

        # -- load paths --
        self.names = read_names(iroot,self.nframes,ext="jpg")
        self.groups = sorted(self.names)

        # -- limit num of samples --
        self.indices = enumerate_indices(len(self.names),
                                         params.nsamples,
                                         params.rand_order,
                                         params.index_skip)
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

        # -- flow io --
        size = list(clean.shape[-2:])
        vid_name = "_".join(vid_name.split("+")[0].split("_")[:-2])
        fflow,bflow = read_flows(FLOW_BASE,self.read_flows,vid_name,
                                 self.noise_info,self.seed,loc,size)
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

        return {'noisy':blur,'clean':sharp,
                'index':index_th,'fnums':frame_nums,'region':region,
                'rng_state':rng_state}

        # return {'blur':blur,'sharp':sharp,
        #         'index':index_th,'fnums':frame_nums,'region':region,
        #         'rng_state':rng_state}

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
    fields = {"batch_size":1,
              "nsamples":-1,
              "isize":None,
              "fstride":1,
              "nframes":0,
              "fskip":1,
              "bw":False,
              "index_skip":1,
              "rand_order":False,
              "cropmode":"rand",
              "read_flows":False,
              "num_workers":2,
              "flippy_augs":None,
              "scale_augs":None,
              "seed":123}
    # fields = {"bw":False,
    #           "nsamples":-1,
    #           "nframes":10,
    #           "fstride":1,
    #           "isize":None,
    #           "nsamples":-1,
    #           "fskip":1,
    #           "index_skip":1,
    #           "batch_size":1,
    #           "rand_order":True,
    #           "cropmode":None}
    p = parse_cfg(cfg,modes,fields)

    # -- setup paths --
    iroot = BASE

    # -- create objcs --
    data = edict()
    tr_set = optional(cfg,"tr_set","train")
    data.tr = GoProCropped(iroot,tr_set,p.tr)
    data.val = GoProCropped(iroot,"test",p.val)
    data.te = GoProCropped(iroot,"test",p.te)

    # -- loaders --
    batch_size = edict({key:val['batch_size'] for key,val in p.items()})
    loader = get_loaders(cfg,data,batch_size)

    return data,loader


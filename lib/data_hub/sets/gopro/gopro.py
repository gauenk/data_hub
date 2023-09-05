"""
GoPro dataset

"""

# -- python imports --
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

# -- optical flow --
import torch as th
from .paths import FLOW_BASE # why not other paths? I think we can do it when time
from data_hub.read_flow import read_flows

# -- local imports --
from .paths import BASE
from .reader import read_files,read_data

class GoPro():

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
        self.fstride = params.fstride

        # -- manage cropping --
        isize,nframes = params.isize,params.nframes
        isize_is_none = isize is None or isize == "none"
        self.crop = params.isize
        self.cropmode = params.cropmode if not(isize_is_none) else "none"
        self.region_temp = None
        if not(isize_is_none):
            self.region_temp = "%d_%d_%d" % (nframes,isize[0],isize[1])

        # # -- load paths --
        # self.names = read_names(iroot,self.nframes,ext="jpg")
        # self.groups = sorted(self.names)

        # -- load paths --
        self.paths = read_files(iroot,split,self.nframes,self.fstride,ext="jpg")
        self.groups = sorted(list(self.paths['images'].keys()))

        # -- limit num of samples --
        self.indices = enumerate_indices(len(self.paths['images']),
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
        vid_files = self.paths['images'][group]
        data = read_data(vid_files,self.bw)

        # -- unpack --
        clean = data.sharp
        noisy = data.blur
        blur_gamma = data.blur_gamma

        # -- meta info --
        frame_nums = th.IntTensor(self.paths['fnums'][group])

        # -- flow io --
        vid_name = group.split(":")[0]
        isize = list(clean.shape[-2:])
        loc = [0,len(clean),0,0]
        noise_info = edict({"ntype":"blur"})
        fflow,bflow = read_flows(FLOW_BASE,self.read_flows,vid_name,
                                 noise_info,self.seed,loc,isize)

        # -- cropping --
        region = th.IntTensor([])
        in_vids = [clean,noisy,fflow,bflow] if self.read_flows else [clean,noisy]
        use_region = "region" in self.cropmode or "coords" in self.cropmode
        if use_region:
            region = crop_vid(clean,self.cropmode,self.isize,self.region_temp)
        else:
            in_vids = crop_vid(in_vids,self.cropmode,self.isize,self.region_temp)
            clean = in_vids[0]
            noisy = in_vids[1]
            if self.read_flows:
                fflow,bflow = in_vids[2],in_vids[3]

        # -- manage flow and output --
        index_th = th.IntTensor([image_index])

        return {'noisy':noisy,'clean':clean,
                'fflow':fflow,'bflow':bflow,
                'blur':noisy,'sharp':clean,'blur_gamma':blur_gamma,
                'index':index_th,'fnums':frame_nums,'region':region,
                'rng_state':rng_state}

        # return {'blur':blur,'sharp':sharp,'blur_gamma':blur_gamma,
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
              "cropmode":"center",
              "num_workers":2,
              "read_flows":False,
              "seed":123}
    p = parse_cfg(cfg,modes,fields)

    # -- setup paths --
    iroot = BASE

    # -- create objcs --
    data = edict()
    data.tr = GoPro(iroot,"train",p.tr)# p.nsamples.tr,p.nframes.tr,
                    # p.fstride.tr,p.isize.tr,p.bw.tr,
                    # p.cropmode.tr,p.rand_order.tr,p.index_skip.tr)
    data.val = GoPro(iroot,"test",p.val)# p.nsamples.val,p.nframes.val,
                     # p.fstride.val,p.isize.val,p.bw.val,
                     # p.cropmode.tr,p.rand_order.val,p.index_skip.val)
    data.te = GoPro(iroot,"test",p.te)# p.nsamples.val,p.nframes.val,
                    # p.fstride.val,p.isize.val,p.bw.val,
                    # p.cropmode.tr,p.rand_order.val,p.index_skip.val)

    # -- create loaders --
    batch_size = edict({key:val['batch_size'] for key,val in p.items()})
    loader = get_loaders(cfg,data,batch_size)

    return data,loader


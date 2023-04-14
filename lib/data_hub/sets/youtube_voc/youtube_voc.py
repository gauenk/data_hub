"""

The YouTubeVOC dataset

"""

# -- python imports --
import pdb,json
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
from data_hub.read_flow import read_flows

# -- local imports --
from .paths import BASE,FLOW_PATH
from .reader import read_files,read_video,read_annos
from .formats import _files_to_dict

class YouTubeVOC():

    def __init__(self,root,split,noise_info,params):

        # -- set init params --
        self.root = root
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
        self.paths = read_files(root,split,params.nframes,
                                params.fstride,ext="png")
        self.groups = sorted(list(self.paths['images'].keys()))

        # -- limit num of samples --
        self.indices = enumerate_indices(len(self.paths['images']),params.nsamples,
                                         params.rand_order,params.index_skip)
        self.nsamples = len(self.indices)

        # -- read meta-data --
        with open(root/split/"meta.json","r") as f:
            self.meta = json.load(f)

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
        vid_name = group
        if ":" in group:
            vid_name = group.split(":")[0]

        # -- load burst --
        vid_files = self.paths['images'][group]
        clean = read_video(vid_files,self.bw)
        clean = th.from_numpy(clean)
        insts,insts_exists = read_annos(vid_files)

        # -- read meta-data --
        labels = self.meta['videos'][vid_name]

        # -- convert --
        annos = _files_to_dict(insts,labels)
        print(annos)

        # -- flow io --
        vid_name = group.split(":")[0]
        isize = list(clean.shape[-2:])
        loc = [0,len(clean),0,0]
        fflow,bflow = read_flows(FLOW_PATH,self.read_flows,vid_name,
                                 self.noise_info,self.seed,loc,isize)

        # -- cropping --
        region = th.IntTensor([])
        in_vids = [clean,insts,fflow,bflow] if self.read_flows else [clean,insts]
        use_region = "region" in self.cropmode or "coords" in self.cropmode
        if use_region:
            region = crop_vid(clean,self.cropmode,self.isize,self.region_temp)
        else:
            in_vids = crop_vid(in_vids,self.cropmode,self.isize,self.region_temp)
            clean,insts = in_vids[0],in_vids[1]
            if self.read_flows:
                fflow,bflow = in_vids[1],in_vids[2]

        # -- get noise --
        # with self.fixRandNoise_1.set_state(index):
        noisy = self.noise_trans(clean)

        # -- manage flow and output --
        index_th = th.IntTensor([image_index])
        frame_nums = th.IntTensor(self.paths['fnums'][group])

        return {'noisy':noisy,'clean':clean,'index':index_th,
                'fnums':frame_nums,'region':region,'rng_state':rng_state,
                'fflow':fflow,'bflow':bflow,"insts":insts,"insts_exists":insts_exists,
                "labels":labels}

#
# Loading the datasets in a project
#

def get_youtubevoc_dataset(cfg):
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

    # -- create objcs --
    data = edict()
    data.tr = YouTubeVOC(BASE,"train",noise_info,p.tr)
    data.val = YouTubeVOC(BASE,"valid",noise_info,p.val)
    data.te = YouTubeVOC(BASE,"test",noise_info,p.val)

    # -- create loaders --
    batch_size = edict({key:val['batch_size'] for key,val in p.items()})
    loader = get_loaders(cfg,data,batch_size)

    return data,loader

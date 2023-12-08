"""
BSD500 dataset

"""

# -- python imports --
import pdb
import numpy as np
from pathlib import Path
from einops import rearrange,repeat
from easydict import EasyDict as edict
import scipy

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
from data_hub.read_flow import read_flows

# -- local imports --
from .paths import IMAGE_PATH,IMAGE_SETS
from PIL import Image
# from .reader import read_video

class BSD500():

    def __init__(self,iroot,split,noise_info,params):

        # -- set init params --
        self.iroot = iroot
        self.split = split
        self.isize = params.isize
        self.bw = params.bw
        self.rand_order = params.rand_order
        self.index_skip = params.index_skip
        self.read_flows = params.read_flows
        self.seed = params.seed
        self.sigma = params.sigma
        self.noise_info = noise_info
        self.chnl4 = params.chnl4

        # -- manage cropping --
        isize = params.isize
        isize_is_none = isize is None or isize == "none"
        self.crop = isize
        self.cropmode = params.cropmode if not(isize_is_none) else "none"
        self.region_temp = None
        if not(isize_is_none):
            self.region_temp = "%d_%d_%d" % (1,isize[0],isize[1])

        # -- create transforms --
        self.noise_trans = get_noise_transform(noise_info,noise_only=True)

        # -- load paths --
        # self.paths = read_files(iroot,split,params.fstride,ext="jpg")
        files = (iroot / split).iterdir()
        prep_it = lambda x: x.name.split(".")[0]
        self.paths = {}
        files = sorted([prep_it(fn) for fn in files if not("Thumbs" in fn.name)])
        self.paths['images'] = files
        self.paths['fnums'] = {fn_id:[0] for fn_id in files}
        self.groups = self.paths['images']

        # -- limit num of samples --
        self.indices = enumerate_indices(len(self.paths['images']),params.nsamples,
                                         params.rand_order,params.index_skip)
        self.nsamples = len(self.indices)
        # print("self.nsamples: ",self.nsamples)

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
        idict = self.paths['images']
        img_fn = self.iroot / self.split / (idict[image_index] + ".jpg")
        clean = Image.open(img_fn).convert("RGB")
        clean = rearrange(th.from_numpy(np.array(clean)),'h w c -> c h w')
        seg_fn = self.iroot.parents[0] / "groundTruth"
        seg_fn = seg_fn / self.split / (idict[image_index] + ".mat")
        info = scipy.io.loadmat(str(seg_fn))
        seg = info['groundTruth']

        # -- meta info --
        frame_nums = th.IntTensor([0])

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

        # -- append 4th channel if necessary --
        if not(self.chnl4 is None):
            if self.chnl4 == "rgbg":
                clean = th.cat([clean,clean[...,[1],:,:]],-3)
            else:
                raise ValueError(f"Uknown chnl4 [{chnl4}]")

        # -- get noise --
        noisy = self.noise_trans(clean)
        sigma = th.FloatTensor([self.sigma])
        if hasattr(self.noise_trans,"sigma"):
            sigma = getattr(self.noise_trans,"sigma")
            sigma = th.FloatTensor([sigma])

        # -- manage flow and output --
        index_th = th.IntTensor([image_index])

        return {'noisy':noisy,'clean':clean,'seg':seg,'index':index_th,
                'fnums':frame_nums,'region':region,'rng_state':rng_state,
                'fflow':fflow,'bflow':bflow,"sigma":sigma}

#
# Loading the datasets in a project
#

def get_bsd500_dataset(cfg):
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
              "fskip":1,
              "bw":False,
              "index_skip":1,
              "rand_order":False,
              "cropmode":"center",
              "num_workers":2,
              "read_flows":False,
              "seed":123,
              "sigma":-1,
              "chnl4":None}
    p = parse_cfg(cfg,modes,fields)

    # -- setup paths --
    iroot = IMAGE_PATH

    # -- create objcs --
    data = edict()
    data.tr = BSD500(iroot,"train",noise_info,p.tr)
    data.val = BSD500(iroot,"val",noise_info,p.val)
    data.te = BSD500(iroot,"test",noise_info,p.te)

    # -- create loaders --
    batch_size = edict({key:val['batch_size'] for key,val in p.items()})
    loader = get_loaders(cfg,data,batch_size)

    return data,loader


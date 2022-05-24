
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
import torch
import torchvision.transforms.functional as tvF

# -- project imports --
from datasets.common import get_loader,optional,get_noise_info
from datasets.transforms import get_noise_transform
from datasets.reproduce import RandomOnce,get_random_state,enumerate_indices

# -- local imports --
from .paths import IMAGE_PATH,IMAGE_SETS
from .reader import read_files,read_subburst_files,read_burst,read_pix,pix2flow

class RawLoader():

    def __init__(self,root,names,split):

        # -- set init params --
        self.iroot = iroot
        self.froot = froot
        self.sroot = sroot
        self.split = split
        self.noise_info = noise_info
        self.ps = ps
        self.nsamples = nsamples
        self.isize = isize
        self.scale = scale
        self.image_set = image_set
        self.read_flow = read_flow
        self.scale = scale # not integrated into "flow" cache
        self.color = color # not integrated into "flow" cache

        # -- create transforms --
        self.noise_trans = get_noise_transform(noise_info,noise_only=True)

        # -- load paths --
        output = read_subburst_files(iroot,froot,sroot,split,isize,ps,
                                     nframes,image_set,read_flow)
        self.paths,self.nframes,all_eq = output

        # self.paths,self.nframes,all_eq = read_files(iroot,froot,sroot,split,
        #                                             isize,ps,nframes)
        # msg = "\n\n\n\nWarning: Not all bursts are same length!!!\n\n\n\n"
        # if not(all_eq): print(msg)
        self.groups = sorted(list(self.paths['images'].keys()))

        # -- limit num of samples --
        self.indices = enumerate_indices(len(self.paths['images']),nsamples)
        self.nsamples = len(self.indices)
        nsamples = self.nsamples
        print("nsamples: ",nsamples)

        # -- single random noise --
        self.noise_once = optional(noise_info,"sim_once",False)
        self.fixRandNoise_1 = RandomOnce(self.noise_once,nsamples)
        self.fixRandNoise_2 = RandomOnce(self.noise_once,nsamples)

    def __len__(self):
        return self.nsamples

    def get_flow(self, group):

        # -- no values if unused --
        if not(self.read_flow):
            size = (1,1,1,1,1)
            return torch.empty(size),torch.empty(size)

        # -- load pix & flow --
        ref_pix = read_pix(self.paths['flows'][group])
        ref_flow = pix2flow(ref_pix)

        # -- format pix --
        ref_pix = rearrange(ref_pix,'two t k h w -> k t h w two')
        ref_pix = torch.LongTensor(ref_pix)#.copy())

        # -- format flow --
        ref_flow = rearrange(ref_flow,'two t k h w -> k t h w two')
        ref_flow = torch.FloatTensor(ref_flow.copy())#.copy())

        return ref_pix,ref_flow

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        # -- get random state --
        rng_state = get_random_state()

        # -- indices --
        image_index = self.indices[index]
        group = self.groups[image_index]
        tframes = len(self.paths['images'][group])
        nframes = tframes if self.nframes is None else self.nframes

        # -- select correct image paths --
        ref = tframes//2
        start = ref - nframes//2
        frame_ids = np.arange(start,start+nframes)

        # -- load burst --
        img_fn = self.paths['images'][group]
        icrop = self.paths['crops'][group]
        dyn_clean = read_burst(img_fn,self.isize,icrop,self.scale,self.color)

        # -- load burst --
        ref_pix,ref_flow = self.get_flow(group)

        # -- get noise --
        with self.fixRandNoise_1.set_state(index):
            dyn_noisy = self.noise_trans(dyn_clean)#+0.5
        nframes,c,h,w = dyn_noisy.shape

        # -- get second, different noise --
        static_clean = repeat(dyn_clean[nframes//2],'c h w -> t c h w',t=nframes)
        with self.fixRandNoise_2.set_state(index):
            static_noisy = self.noise_trans(static_clean)#+0.5

        # -- manage flow and output --
        index_th = torch.IntTensor([image_index])

        return {'dyn_noisy':dyn_noisy,'dyn_clean':dyn_clean,
                'static_noisy':static_noisy,'static_clean':static_clean,
                'nnf':ref_flow,'seq_flow':None, 'ref_flow':ref_flow,
                'flow':ref_flow,'index':index_th,'rng_state':rng_state,
                'ref_pix':ref_pix}


#
# Loading the datasets in a project
#

def get_davis_dataset(cfg,image_set="full"):

    #
    # -- extract --
    #

    # -- noise and dyanmics --
    noise_info = get_noise_info(cfg)
    isize = optional(cfg,"frame_size",None)
    nframes = optional(cfg,"nframes",None)
    ps = optional(cfg,"patchsize",1)
    read_flow = optional(cfg,"read_flow",False)
    scale = optional(cfg,"scale",1.0)
    color = optional(cfg,"color",True)

    # -- samples --
    nsamples = optional(cfg,"nsamples",0)
    tr_nsamples = optional(cfg,"tr_nsamples",nsamples)
    val_nsamples = optional(cfg,"val_nsamples",nsamples)
    te_nsamples = optional(cfg,"te_nsamples",nsamples)

    # -- setup paths --
    iroot = IMAGE_PATH
    froot = FLOW_PATH

    # -- image sets path --
    if image_set == "full":
        sroot = IMAGE_SETS
    elif image_set == "small":
        sroot = IMAGE_SETS_SMALL
    elif image_set == "tiny":
        sroot = IMAGE_SETS_TINY
    else:
        raise ValueError(f"Uknown image set [{image_set}]")

    # -- create objcs --
    data = edict()
    data.tr = DAVIS(iroot,froot,sroot,"train",isize,scale,ps,color,
                    tr_nsamples,nframes,noise_info,image_set,read_flow)
    data.val = DAVIS(iroot,froot,sroot,"val",isize,scale,ps,color,
                     val_nsamples,nframes,noise_info,image_set,read_flow)
    data.te = DAVIS(iroot,froot,sroot,"test",isize,scale,ps,color,
                    te_nsamples,nframes,noise_info,image_set,read_flow)

    # -- create loader --
    loader = get_loader(cfg,data,cfg.batch_size)

    return data,loader

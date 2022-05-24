"""
Toy64 dataset

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
from data_hub.common import get_loaders,optional
from data_hub.transforms import get_noise_transform
from data_hub.reproduce import RandomOnce,get_random_state,enumerate_indices

# -- local imports --
from .paths import IMAGE_PATH,IMAGE_SETS
from .reader import read_files,read_video,read_pix,pix2flow

class Toy64():

    def __init__(self,iroot,sroot,split,noise_info,nsamples=0):

        # -- set init params --
        self.iroot = iroot
        self.sroot = sroot
        self.split = split

        # -- create transforms --
        self.noise_trans = get_noise_transform(noise_info,noise_only=True)

        # -- load paths --
        self.paths = read_files(iroot,sroot,split)
        self.groups = sorted(list(self.paths['images'].keys()))

        # -- limit num of samples --
        self.indices = enumerate_indices(len(self.paths['images']),nsamples)
        self.indices = len(self.paths['images']))
        self.nsamples = len(self.indices)

        # -- repro --
        self.noise_once = optional(noise_info,"sim_once",False)
        self.fixRandNoise_1 = RandomOnce(self.noise_once,self.nsamples)

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
        vid_files = self.paths['images'][group]
        clean = read_video(files)

        # -- get noise --
        with self.fixRandNoise_1.set_state(index):
            noisy = self.noise_trans(dyn_clean)

        # -- manage flow and output --
        index_th = torch.IntTensor([image_index])

        return {'noisy':dyn_noisy,'clean':dyn_clean,'index':index_th,
                'rng_state':rng_state}

#
# Loading the datasets in a project
#

def get_toy64_dataset(cfg):

    #
    # -- extract --
    #

    # -- noise and dyanmics --
    noise_info = create_noise_info_from_cfg(cfg)

    # -- samples --
    nsamples = optional(cfg,"nsamples",0)
    tr_nsamples = optional(cfg,"tr_nsamples",nsamples)
    val_nsamples = optional(cfg,"val_nsamples",nsamples)
    te_nsamples = optional(cfg,"te_nsamples",nsamples)

    # -- setup paths --
    iroot = IMAGE_PATH
    sroot = IMAGE_SETS

    # -- create objcs --
    data = edict()
    data.tr = Toy64(iroot,sroot,"train",noise_info,tr_nsamples)
    data.val = Toy64(iroot,sroot,"train",noise_info,val_nsamples)
    data.te = Toy64(iroot,sroot,"train",noise_info,te_nsamples)

    # -- create loader --
    batch_size = optional(cfg,'batch_size',1)
    loader = get_loaders(cfg,data,batch_size)

    return data,loader



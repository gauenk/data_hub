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
from datasets.common import get_loaders,optional,get_noise_info

# -- local imports --
from .paths import IMAGE_PATH,IMAGE_SETS
from .reader import read_set,read_files,read_burst,read_chunked_files

class SID():

    def __init__(self,iroot,ds_set,nframes):

        # -- set init params --
        self.iroot = iroot
        self.unique_vid_ids = read_set(IMAGE_SETS/("%s.txt" % ds_set))

        # -- load paths --
        self.paths = read_chunked_files(iroot,self.unique_vid_ids,nframes)
        self.groups = sorted(list(self.paths['images'].keys()))

        # -- limit num of samples --
        self.nsamples = len(self.groups)
        self.indices = np.arange(self.nsamples)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        # -- indices --
        image_index = self.indices[index]
        group = self.groups[image_index]
        tframes = len(self.paths['images'][group])
        nframes = tframes if self.nframes is None else self.nframes

        # -- load burst --
        img_fn = self.paths['ref'][group]
        clean = read_burst(img_fn)

        # -- load burst --
        ref_pix,ref_flow = self.get_flow(group)

        # -- manage flow and output --
        index_th = torch.IntTensor([image_index])

        return {'noisy':noisy,'clean':clean,"index":index_th}


#
# Loading the datasets in a project
#

def get_sid_dataset(cfg):

    # -- setup paths --
    iroot = IMAGE_PATH

    # -- create objcs --
    data = edict()
    data.tr = SID(iroot,"train",nframes)
    data.val = SID(iroot,"val",nframes)
    data.te = SID(iroot,"test",nframes)

    # -- create loader --
    loaders = get_loaders(cfg,data,cfg.batch_size)

    return data,loaders



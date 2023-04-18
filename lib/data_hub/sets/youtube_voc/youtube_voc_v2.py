"""

The YouTubeVOC dataset

"""

# -- python imports --
import pdb,json,copy
dcopy = copy.deepcopy
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

# -- detectron2 --
import operator
try:
    import detectron2.data.transforms as T
    from .mapper import DatasetMapperSeq
except:
    pass

# -- optical flow --
import torch as th
from data_hub.read_flow import read_flows

# -- local imports --
from .paths import BASE,FLOW_PATH
from .reader import read_files,read_video,read_annos
from .formats import cached_format

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

def get_augs(is_train,isize):
    min_size = 128
    max_size = 1024
    sample_style = "choice"
    random_flip = "horizontal"
    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    if is_train:# and cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(prob=0.9,horizontal=random_flip=="horizontal",
                         vertical=random_flip=="vertical")
        )
    if not(isize is None):
        cH,cW = isize
        augmentation.insert(0,T.RandomCrop("absolute", [cH,cW]))
    return augmentation

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
        self.paths = read_files(root,split,0,1,ext="png")
        self.groups = sorted(list(self.paths['images'].keys()))

        # -- read meta-data --
        with open(root/split/"meta.json","r") as f:
            self.meta = json.load(f)

        # -- labels to ints --
        cats = ["None",]+np.loadtxt(root/"cats.txt",dtype=str).tolist()
        cats_ids = np.arange(len(cats)).tolist()
        cats = dict(zip(cats,cats_ids))

        # -- process --
        self.annos = cached_format(root,split,self.paths['images'],
                                   self.paths['annos'],self.meta,cats,to_polygons=True)

        # -- mapper --
        self.nframes = params.nframes
        is_train = split=="train"
        augs = get_augs(is_train,isize)
        image_format = "RGB"
        self.mapper = DatasetMapperSeq(is_train=is_train,
                                       augmentations=augs,
                                       image_format=image_format)

        # -- num of samples --
        self.input_nsamples = params.nsamples
        self.rand_order = params.rand_order
        self.index_skip = params.index_skip
        self.reset_sample_indices()

    def reset_sample_indices(self):
        # -- limit num of samples --
        self.indices = enumerate_indices(len(self.paths['images']),
                                         self.input_nsamples,
                                         self.rand_order,self.index_skip)
        self.nsamples = len(self.indices)

        # # -- repro --
        # self.noise_once = optional(noise_info,"sim_once",False)
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

        # -- indices --
        image_index = self.indices[index].item()

        # -- use mapper --
        fmted = self.mapper(self.annos[image_index],self.nframes)

        # # -- flow io --
        # vid_name = group.split(":")[0]
        # isize = list(clean.shape[-2:])
        # loc = [0,len(clean),0,0]
        # fflow,bflow = read_flows(FLOW_PATH,self.read_flows,vid_name,
        #                          self.noise_info,self.seed,loc,isize)
        fmted['image_index'] = image_index
        fmted['fflow'] = th.tensor([0])
        fmted['bflow'] = th.tensor([0])
        # print("__getitem__: ",type(fmted['instances'][0]))
        return fmted

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
              "num_workers":2,
              "read_flows":False,
              "seed":123}
    p = parse_cfg(cfg,modes,fields)

    # -- create objcs --
    data = edict()
    data.tr = YouTubeVOC(BASE,"train",noise_info,p.tr)
    data.val = YouTubeVOC(BASE,"valid",noise_info,p.val)
    data.te = data.val

    # -- create loaders --
    batch_size = edict({key:val['batch_size'] for key,val in p.items()})
    # cfg.collate_fn = operator.itemgetter(0)
    cfg.collate_fn = trivial_batch_collator
    loader = get_loaders(cfg,data,batch_size)

    # # -- build training dataset --
    # mapper = DatasetMapper(cfg, is_train=split=="train",
    #                        augmentations=build_sem_seg_train_aug(cfg))
    # loaders.tr = build_detection_train_loader(cfg, mapper=mapper) # the data loader.



    return data,loader

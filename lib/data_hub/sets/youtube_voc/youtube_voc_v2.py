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

# -- datahub --
import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
from detectron2.data import DatasetMapperSeq
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.point_rend import ColorAugSSDTransform, add_pointrend_config
from detectron2.data import transforms as T
import operator

# -- optical flow --
import torch as th
from data_hub.read_flow import read_flows

# -- local imports --
from .paths import BASE,FLOW_PATH
from .reader import read_files,read_video,read_annos
from .formats import cached_format



# def build_batch_data_loader(
#     dataset,
#     sampler,
#     total_batch_size,
#     *,
#     aspect_ratio_grouping=False,
#     num_workers=0,
#     collate_fn=None,
# ):
#     """
#     Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
#     1. support aspect ratio grouping options
#     2. use no "batch collation", because this is common for detection training

#     Args:
#         dataset (torch.utils.data.Dataset): a pytorch map-style or iterable dataset.
#         sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces indices.
#             Must be provided iff. ``dataset`` is a map-style dataset.
#         total_batch_size, aspect_ratio_grouping, num_workers, collate_fn: see
#             :func:`build_detection_train_loader`.

#     Returns:
#         iterable[list]. Length of each list is the batch size of the current
#             GPU. Each element in the list comes from the dataset.
#     """
#     if aspect_ratio_grouping:
#         data_loader = torchdata.DataLoader(
#             dataset,
#             num_workers=num_workers,
#             collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
#             worker_init_fn=worker_init_reset_seed,
#         )  # yield individual mapped dict
#         data_loader = AspectRatioGroupedDataset(data_loader, batch_size)
#         if collate_fn is None:
#             return data_loader
#         return MapDataset(data_loader, collate_fn)
#     else:
#         return torchdata.DataLoader(
#             dataset,
#             batch_size=batch_size,
#             drop_last=True,
#             num_workers=num_workers,
#             collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
#             worker_init_fn=worker_init_reset_seed,
#         )

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def get_augs(is_train,isize):
    # cfg = edict()
    min_size = 128
    max_size = 1024
    sample_style = "choice"
    random_flip = "horizontal"
    # if is_train:
    #     # min_size = cfg.INPUT.MIN_SIZE_TRAIN
    #     # max_size = cfg.INPUT.MAX_SIZE_TRAIN
    #     # sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    # else:
    #     # min_size = cfg.INPUT.MIN_SIZE_TEST
    #     # max_size = cfg.INPUT.MAX_SIZE_TEST
    #     sample_style = "choice"
    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]

    if is_train:# and cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(prob=0.9,horizontal=random_flip=="horizontal",
                         vertical=random_flip=="vertical")
        )
    if not(isize is None):
        # print(isize)
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
        self.paths = read_files(root,split,params.nframes,
                                params.fstride,ext="png")
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
        # is_train: bool,
        # *,
        # augmentations: List[Union[T.Augmentation, T.Transform]],
        # image_format: str,
        # use_instance_mask: bool = False,
        # use_keypoint: bool = False,
        # instance_mask_format: str = "polygon",
        # keypoint_hflip_indices: Optional[np.ndarray] = None,
        # precomputed_proposal_topk: Optional[int] = None,
        # recompute_boxes: bool = False,

        is_train = split=="train"
        augs = get_augs(is_train,isize)
        image_format = "RGB"
        self.mapper = DatasetMapperSeq(is_train=is_train,
                                       augmentations=augs,
                                       image_format=image_format)

        # # -- limit num of samples --
        # self.indices = enumerate_indices(len(self.paths['images']),params.nsamples,
        #                                  params.rand_order,params.index_skip)
        # self.nsamples = len(self.indices)
        ns = params.nsamples
        self.nsamples = ns if ns > 0 else len(self.annos)
        print("self.nsamples: ",self.nsamples,split)

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

        # -- use mapper --
        fmted = self.mapper(self.annos[index])
        # # -- flow io --
        # vid_name = group.split(":")[0]
        # isize = list(clean.shape[-2:])
        # loc = [0,len(clean),0,0]
        # fflow,bflow = read_flows(FLOW_PATH,self.read_flows,vid_name,
        #                          self.noise_info,self.seed,loc,isize)
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
    data.val = YouTubeVOC(BASE,"train",noise_info,p.val)
    data.te = YouTubeVOC(BASE,"train",noise_info,p.te)

    # data.val = YouTubeVOC(BASE,"valid",noise_info,p.val)
    # data.te = YouTubeVOC(BASE,"test",noise_info,p.val)

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

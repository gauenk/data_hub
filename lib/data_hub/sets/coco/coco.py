
# -- detectron2 --
try:
    from detectron2.config import instantiate
    from detectron2.data.transforms.augmentation_impl import ResizeShortestEdge
    from detectron2.data import (
        DatasetMapper,
        build_detection_test_loader,
        build_detection_train_loader,
        get_detection_dataset_dicts,
    )
    from detectron2.evaluation import COCOEvaluator
except:
    pass


# -- project imports --
from data_hub.common import get_loaders,optional,get_isize
from data_hub.transforms import get_noise_transform,noise_from_cfg
from data_hub.reproduce import RandomOnce,get_random_state,enumerate_indices
from data_hub.cropping import crop_vid
from data_hub.opt_parsing import parse_cfg

# -- python imports --
import pdb
import numpy as np
from pathlib import Path
from einops import rearrange,repeat
from easydict import EasyDict as edict
from .paths import BASE,IMAGE_PATH,IMAGE_SETS

def get_test_config(year,dset):
    dset_name = "coco_%s_%s" % (year,dset)
    image_format = "RGB"
    cfg = {'dataset': {'names': dset_name, 'filter_empty': False,
                       '_target_': get_detection_dataset_dicts},
           'mapper': {'is_train': False, 'augmentations':
                      [{'short_edge_length': 1024, 'max_size': 1024,
                        '_target_': ResizeShortestEdge}],
                      'image_format': image_format,
                      '_target_': DatasetMapper},
           'num_workers': 4,
           '_target_': build_detection_test_loader}
    return cfg

def get_eval_config(year,dset):
    dset_name = "coco_%s_%s" % (year,dset)
    cfg = {'dataset_name': dset_name,
           '_target_': COCOEvaluator}
    return cfg

class COCODataHub():

    def __init__(self,iroot,sroot,split,cfg):
        self.split = split
        self.iroot = iroot
        self.sroot = sroot
        self.year = cfg.coco_year
        self.dset_te = cfg.coco_dset_te
        test_cfg = get_test_config(self.year,self.dset_te)
        eval_cfg = get_eval_config(self.year,self.dset_te)
        self.tester = instantiate(test_cfg)
        self.evals = instantiate(eval_cfg)

    def __len__(self):
        return self.coco_dset.nsamples

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
        frame_nums = th.IntTensor(self.paths['fnums'][group])

        # -- cropping --
        region = th.IntTensor([])
        use_region = "region" in self.cropmode or "coords" in self.cropmode
        if use_region:
            region = crop_vid(clean,self.cropmode,self.isize,self.region_temp)
        else:
            clean = crop_vid([clean],self.cropmode,self.isize,self.region_temp)[0]

        # -- get noise --
        # with self.fixRandNoise_1.set_state(index):
        noisy = self.noise_trans(clean)

        # -- manage flow and output --
        index_th = th.IntTensor([image_index])

        return {'noisy':noisy,'clean':clean,'index':index_th,
                'fnums':frame_nums,'region':region,'rng_state':rng_state}

#
# Loading the datasets in a project
#

def get_coco_dataset(cfg):
    return load(cfg)


def load(cfg):

    #
    # -- extract --
    #

    # -- field names and defaults --
    modes = ["tr","val","te"]
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
              "seed":123,
              "coco_year":"2017",
              "coco_dset_tr":"tr",
              "coco_dset_te":"val"}
    p = parse_cfg(cfg,modes,fields)

    # -- setup paths --
    iroot = IMAGE_PATH
    sroot = IMAGE_SETS

    # -- create objcs --
    data = edict()
    # data.tr = COCODataHub(iroot,sroot,p.tr.coco_year,
    #                       p.tr.coco_dset_tr,p.tr.coco_dset_te)
    data.tr = COCODataHub(iroot,sroot,"tr",p.tr)
    data.te = COCODataHub(iroot,sroot,"te",p.te)


    # -- create loaders --
    batch_size = edict({key:val['batch_size'] for key,val in p.items()})
    loader = get_loaders(cfg,data,batch_size)

    return data,loader



import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch
from einops import rearrange

# from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

class DatasetMapperSeq:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    # @configurable
    def __init__(
        self,
        is_train: bool,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations) # todo: video
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    def _transform_annotations(self, dataset_dict, transforms, image_shape, NF):
        # # USER: Modify this if you want to keep them for some reason.
        # for anno in dataset_dict["annotations"]:
        #     if not self.use_instance_mask:
        #         anno.pop("segmentation", None)
        #     if not self.use_keypoint:
        #         anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = dataset_dict['annotations']
        instances = []
        for t in range(NF):
            annos_t = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in annos[t]
            ]
            # NOTE: deleted the "crowded filter"
            instances_t = utils.annotations_to_instances(
                annos_t, image_shape, mask_format=self.instance_mask_format
            )

            # recompute bbox after transform
            if self.recompute_boxes:
                instances_t.gt_boxes = instances_t.gt_masks.get_bounding_boxes()
            instances_t = utils.filter_empty_instances(instances_t,by_box=False)
            # print(instances_t)
            # print(type(instances_t))
            instances.append(instances_t)
        dataset_dict["instances"] = instances

    def __call__(self,dataset_dict,nframes):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # USER: Write your own image loading if it's not from a file
        vid = []
        fns = dataset_dict["file_name"]
        nframes = nframes if nframes > 0 else len(fns)
        NF = min(len(fns),nframes)
        for t in range(NF):
            vid.append(utils.read_image(fns[t], format=self.image_format))
        # utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        sem_seg = []
        if "sem_seg_file_name" in dataset_dict:
            fns = dataset_dict["sem_seg_file_name"]
            for t in range(NF):
                sem_seg.append(utils.read_image(fns[t], format="L").squeeze(2))
        else:
            sem_seg = None

        # -------------------------------
        #         augmentation
        # -------------------------------

        # -- augment [t == 0] --
        aug_input = T.AugInput(vid[0], sem_seg=sem_seg[0])
        transforms = self.augmentations(aug_input)
        # print(transforms)
        vid[0], sem_seg[0] = aug_input.image, aug_input.sem_seg
        # print("NF: ",NF)

        # -- apply same transform to video --
        for t in range(1,NF):
            aug_input = T.AugInput(vid[t], sem_seg=sem_seg[t])
            aug_input.transform(transforms)
            vid[t], sem_seg[t] = aug_input.image, aug_input.sem_seg

        # -- stack --
        vid = np.stack(vid)
        vid = rearrange(vid,'t h w c -> t c h w')
        sem_seg = np.stack(sem_seg)
        # print(vid.shape,sem_seg.shape)
        image_shape = vid[0].shape[-2:]  # h, w

        # -- to torch --
        dataset_dict["video"] = torch.as_tensor(np.ascontiguousarray(vid))
        if sem_seg is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        # if self.proposal_topk is not None:
        #     utils.transform_proposals(
        #         dataset_dict, image_shape, transforms,
        #         proposal_topk=self.proposal_topk
        #     )

        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     dataset_dict.pop("sem_seg_file_name", None)
        #     return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape, NF)
        # print("dataset_dict.keys(): ",list(dataset_dict.keys()))

        # -- limit to nframes --
        fields = ["annotations","file_name"]
        for field in fields:
            vals = []
            for t in range(NF): vals.append(dataset_dict[field][t])
            dataset_dict[field] = vals

        return dataset_dict


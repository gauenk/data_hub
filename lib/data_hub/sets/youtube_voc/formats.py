
import os,cv2,tqdm
import numpy as np
import torch as th
from PIL import Image

from detectron2.structures import BoxMode
from .reader import read_annos

import cache_io

def cached_format(root,split,img_paths,anno_paths,labels,cats,
                  to_polygons=False,reset=False):

    # -- check cache --
    cache_name = str(root / (".cache/format_%s" % split))
    pkl_name = str(root / (".cache_pkl/format_%s.pkl" % split))
    cache = cache_io.ExpCache(cache_name)
    # reset = True
    if reset: cache.clear()
    # reload = False or reset
    # records = cache.to_records_fast(pkl_name,clear=reload)
    # print(len(records),len(img_paths))
    # # _,_,results = cache.load_raw_fast()
    # if len(records) == len(img_paths):
    #     return records.to_dict()
    verbose = False

    # -- run --
    MAX = 100
    cnt = 0
    keys = list(anno_paths.keys())
    results = []
    for group_name in tqdm.tqdm(keys,disable=not(verbose)):
        cfg = {"group_name":group_name}
        uuid = cache.get_uuid(cfg)
        results_g = cache.read_results(uuid)
        if results_g is None:
            results_g = _files_to_dict(img_paths[group_name],
                                       anno_paths[group_name],
                                       labels['videos'][group_name]['objects'],
                                       group_name,cats,to_polygons=to_polygons)
            cache.save_exp(uuid,cfg,results_g)
        results.append(results_g)
        cnt += 1
        if cnt > MAX:
            break

    return results

def _files_to_dict(img_paths,anno_paths,labels,group_name,cats,to_polygons=False):
    """
    Parse cityscapes annotation files to a instance segmentation dataset dict.

    Args:
        files (tuple): consists of (image_file, instance_id_file, label_id_file, json_file)
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        A dict in Detectron2 Dataset format.
    """
    # from cityscapesscripts.helpers.labels import id2label, name2label
    # image_file, instance_id_file, _, json_file = files
    annos = []
    W,H = Image.open(str(img_paths[0])).size
    insts_vid,_ = read_annos(anno_paths,H,W)
    insts_ids = np.unique(insts_vid.cpu().numpy())

    ret = {
        "file_name": img_paths,
        "image_id": group_name,
        "height": insts_vid.shape[-2],
        "width": insts_vid.shape[-1],
        "sem_seg_file_name": anno_paths,
    }

    T = insts_vid.shape[0]
    for t in range(T):
        annos_t = []
        insts_ids_t = np.unique(insts_vid[t].cpu().numpy())
        for insts_id in insts_ids_t:
            if insts_id == 0: continue
            mask = np.asarray(insts_vid[t] == insts_id, dtype=np.uint8, order="F")
            anno = {}
            cat = labels[str(insts_id)]['category']
            cat = "None" if cat is None else cat
            anno["category_id"] = cats[cat]
            inds = np.nonzero(mask)
            ymin, ymax = inds[0].min(), inds[0].max()
            xmin, xmax = inds[1].min(), inds[1].max()
            anno["bbox"] = (xmin, ymin, xmax, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue
            anno["bbox_mode"] = BoxMode.XYXY_ABS

            # -- to polygons --
            if to_polygons:
                contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)[-2]
                polygons = [c.reshape(-1).tolist() for c in contours if len(c) >= 3]
                # opencv's can produce invalid polygons
                if len(polygons) == 0:
                    continue
                anno["segmentation"] = polygons
            else:
                anno["segmentation"] = mask_util.encode(mask[:, :, None])[0]
            annos_t.append(anno)
        annos.append(annos_t)

    ret["annotations"] = annos
    return ret

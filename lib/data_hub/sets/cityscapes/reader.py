

# -- read data --
import torch as th
import numpy as np
from PIL import Image
from pathlib import Path
from einops import rearrange,repeat

# -- read paths --
import cache_io
import glob
import numpy as np


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#             Read Data
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def read_video(paths,bw=False):
    vid = []
    print(len(paths))
    for path_t in paths:
        if not Path(path_t).exists(): break
        vid_t = Image.open(path_t)
        if bw: vid_t = np.array(vid_t.convert("L"))[...,None]
        else: vid_t = np.array(vid_t.convert("RGB"))
        vid_t = vid_t.astype(np.float32)
        vid_t = rearrange(vid_t,'h w c -> c h w')
        vid.append(vid_t)
    vid = np.stack(vid).astype(np.float32)
    return vid

def read_annos(ipath,atype):
    apath = Path(ipath.replace("leftImg8bit_sequence",f"annos/{atype}"))
    adir = apath.parents[0]
    first = "_".join(apath.name.split("_")[:2])
    last = "_".join(apath.name.split("_")[-2:])
    suffixes = ["color","instanceIds","labelIds"]
    annos = {}
    for suffix in suffixes:
        anno_fn = "%s_000019_%s_%s.png" % (first,atype,suffix)
        anno_img = np.array(Image.open(str(adir/anno_fn)))
        annos[suffix] = th.from_numpy(anno_img)
    return annos

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#             Read Files
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def get_video_names(sroot,region_name):
    region_dir = sroot / region_name
    vid_names = [v.name.split("_")[1] for v in region_dir.iterdir()]
    return np.unique(vid_names)

def get_video_paths(sroot,region_name,vid_name,ext="png"):
    region_dir = sroot / region_name
    srch = "%s_%s_*_leftImg8bit.%s" % (region_name,vid_name,ext)
    subvids = list(glob.glob(str(region_dir/srch)))
    fnums = [int(Path(s).name.split("_")[2]) for s in subvids]
    # print(subvids)
    # subvids = sorted(subvids,key=lambda x: fnums[x])
    subvids = [x for _, x in sorted(zip(fnums,subvids))]
    fnums = sorted(fnums)
    # print(subvids)
    return subvids,fnums

def read_files(iroot,split,nframes,stride,only_anno,ext="png",reset=False):

    # -- check cache --
    args = (split,nframes,stride,only_anno)
    cache_name = str(iroot / (".cache/reader_%s_%s_%s_%s" % args))
    cache = cache_io.ExpCache(cache_name)
    if reset: cache.clear()
    results = cache.read_results("0")
    if not(results is None):
        return results['paths']

    # -- metadata --
    ANNO_FRAME = 19

    # -- not input yet --
    dil = 1

    # -- get vid names in set --
    sroot = iroot/split
    region_names = sorted([s.name for s in sroot.iterdir()])

    # -- get files --
    files = {'images':{},"fnums":{}}
    for region_name in region_names:
        vid_names = get_video_names(sroot,region_name)
        for vid_name in vid_names:
            vid_paths,frame_nums = get_video_paths(sroot,region_name,vid_name,ext)
            total_nframes = len(vid_paths)
            assert total_nframes > 0

            # -- pick number of sub frames --
            nframes_vid = nframes
            if nframes_vid <= 0:
                  nframes_vid = total_nframes

            # -- compute num subframes --
            n_subvids = (total_nframes - (nframes_vid-1)*dil - 1)//stride + 1

            # -- reflect bound --
            def bnd(num,lim):
                if num >= lim: return 2*(lim-1)-num
                else: return num

            vid_id_base = "%s-%s" % (region_name,vid_name)
            for group in range(n_subvids):
                start_t = group * stride
                if n_subvids > 1: vid_id = "%s:%d" % (vid_id_base,start_t)
                else: vid_id = vid_id_base
                end_t = start_t + nframes_vid
                paths_t = [vid_paths[bnd(t,total_nframes)] for t in range(start_t,end_t)]
                fnums_t = [frame_nums[bnd(t,total_nframes)] for t in range(start_t,end_t)]

                # -- optionally only keep vids with the annotation --
                anno_within = ANNO_FRAME in list(fnums_t)
                # print(ANNO_FRAME,fnums_t,only_anno,anno_within)
                if only_anno and not(anno_within): continue

                # -- append --
                files['images'][vid_id] = paths_t
                files['fnums'][vid_id] = fnums_t


    # -- save cache --
    cache.save_exp("0","0",{"paths":files})

    return files


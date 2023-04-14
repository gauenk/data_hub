

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

def read_annos(ipaths):
    annos = []
    for ipath in ipaths:
        apath = Path(str(ipath).replace("JPEGImages","Annotations"))
        apath = str(apath.parent / apath.name.replace("jpg","png"))
        anno_img = np.array(Image.open(apath))
        anno_img = th.from_numpy(anno_img)
        annos.append(anno_img)
    annos = th.stack(annos)
    return annos

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#             Read Files
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def get_video_paths(root,ext="jpg"):
    fns = sorted(list(root.iterdir()))
    fnums = [int(f.stem) for f in fns]
    return fns,fnums

def read_files(root,split,nframes,stride,ext="jpg",reset=False):

    # -- check cache --
    args = (split,nframes,stride)
    cache_name = str(root / (".cache/reader_%s_%s_%s" % args))
    cache = cache_io.ExpCache(cache_name)
    if reset: cache.clear()
    results = cache.read_results("0")
    if not(results is None):
        return results['paths']

    # -- not input yet --
    dil = 1

    # -- get vid names in set --
    iroot = root/split/"JPEGImages"
    vid_names = sorted([s.name for s in iroot.iterdir()])

    # -- get files --
    files = {'images':{},"fnums":{}}
    for vid_name in vid_names:

        # -- get video paths --
        vid_paths,frame_nums = get_video_paths(iroot/vid_name,ext)
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

        for group in range(n_subvids):
            start_t = group * stride
            if n_subvids > 1: vid_id = "%s:%d" % (vid_name,start_t)
            else: vid_id = vid_name
            end_t = start_t + nframes_vid

            # -- unpack --
            paths_t = [vid_paths[bnd(t,total_nframes)] for t in range(start_t,end_t)]
            fnums_t = [frame_nums[bnd(t,total_nframes)] for t in range(start_t,end_t)]

            # -- append --
            files['images'][vid_id] = paths_t
            files['fnums'][vid_id] = fnums_t


    # -- save cache --
    cache.save_exp("0","0",{"paths":files})

    return files


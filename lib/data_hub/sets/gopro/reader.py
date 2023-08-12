
import torch as th
import numpy as np
from PIL import Image
from pathlib import Path
from easydict import EasyDict as edict
from einops import rearrange,repeat

def read_video_in_dir(ipath,nframes,ext="png"):
    vid = []
    for t in range(nframes):
        path_t = path / ("%05d.%s" % (t,ext))
        if not path_t.exists(): break
        vid_t = Image.open(str(path_t)).convert("RGB")
        vid_t = np.array(vid_t)*1.
        vid_t = rearrange(vid_t,'h w c -> c h w')
        vid.append(vid_t)
    vid = np.stack(vid)
    return vid

def read_data(paths_blur,bw=False):

    # -- get files --
    paths = edict()
    paths.blur = paths_blur
    paths.blur_gamma = [Path(str(p).replace("blur","blur_gamma")) for p in paths_blur]
    paths.sharp = [Path(str(p).replace("blur","sharp")) for p in paths_blur]

    # -- read video --
    data = edict()
    for key in paths:
        data[key] = th.from_numpy(read_video(paths[key],bw))
    return data

def read_video(paths,bw=False):
    vid = []
    for path_t in paths:
        if not path_t.exists(): break
        vid_t = Image.open(str(path_t))
        if bw: vid_t = np.array(vid_t.convert("L"))[...,None]
        else: vid_t = np.array(vid_t.convert("RGB"))
        vid_t = vid_t.astype(np.float32)
        vid_t = rearrange(vid_t,'h w c -> c h w')
        vid.append(vid_t)
    vid = np.stack(vid).astype(np.float32)
    return vid

def get_video_paths(vid_dir,ext="png"):

    # -- load all file paths --
    vid_fns,vid_ids = [],[]
    for vid_fn in vid_dir.iterdir():
        vid_id = int(vid_fn.stem.split(".")[0])
        vid_ids.append(vid_id)
        vid_fns.append(vid_fn)

    # -- reorder --
    order = np.argsort(vid_ids)
    vid_ids = [vid_ids[o] for o in order]
    vid_fns = [vid_fns[o] for o in order]

    # -- frame ids start @ zero --
    vid_ids = np.array(vid_ids)
    vid_ids -= vid_ids.min()
    vid_ids = list(vid_ids)

    return vid_fns,vid_ids

def get_vid_names(sroot):
    vid_names = []
    for vid_name in sroot.iterdir():
        if vid_name.stem in ["input","target","groundtruth"]: continue
        vid_names.append(str(vid_name.stem))
    vid_names = sorted(vid_names)
    nvids = len(vid_names)
    # vid_names_abbr = ["%02d" % vidx for vidx in range(nvids)]
    vid_names_abbr = vid_names
    return vid_names,vid_names_abbr

def read_files(iroot,ds_split,nframes,stride,ext="png"):
    """

    what is stride?

    stride = 1
    _ _ _ _ _ _ _
    x x x
      x x x
        x x x

    stride = 2
    _ _ _ _ _ _ _
    x x x
        x x x
            x x x

    what is dilation?

    dilation = 2
    _ _ _ _ _ _ _
    x   x   x
        x   x   x

    """

    # -- not input yet --
    dil = 1

    # -- get vid names in set --
    sroot = iroot / ds_split
    assert sroot.exists()
    vid_names,vid_names_abbr = get_vid_names(sroot)

    # -- get files --
    files = {'images':{},"fnums":{},'names':{}}
    for vid_name,vid_name_abbr in zip(vid_names,vid_names_abbr):
        vid_dir = sroot/vid_name/"blur"
        vid_paths,frame_nums = get_video_paths(vid_dir,ext)
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
            if n_subvids == 1: vid_id = vid_name_abbr
            else: vid_id = "%s:%d" % (vid_name_abbr,start_t)
            end_t = start_t + nframes_vid
            paths_t = [vid_paths[bnd(t,total_nframes)] for t in range(start_t,end_t)]
            files['images'][vid_id] = paths_t

            # -- extra --
            fnums_t = [frame_nums[bnd(t,total_nframes)] for t in range(start_t,end_t)]
            files['fnums'][vid_id] = fnums_t

            # -- full names --
            files['names'][vid_id] = vid_name

    return files


import torch as th
import numpy as np
from PIL import Image
from pathlib import Path
from easydict import EasyDict as edict
from einops import rearrange,repeat

# def read_data(paths_clean,bw=False):

#     # -- get files --
#     paths = edict()
#     paths.blur = paths_clean
#     paths.blur_gamma = [Path(str(p).replace("blur","blur_gamma")) for p in paths_clean]
#     paths.sharp = [Path(str(p).replace("blur","sharp")) for p in paths_clean]

#     # -- read video --
#     data = edict()
#     for key in paths:
#         data[key] = th.from_numpy(read_video(paths[key],bw))
#     return data

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
    vid = th.from_numpy(vid)
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

def get_vid_names(vid_fn):
    with open(vid_fn,"r") as f:
        names = f.readlines()
    names = [name.strip() for name in names]
    return names

def read_names(iroot,sroot,nframes,ds_split,ext="png"):
    """
    Just read the folder names
    """

    # -- read split names --
    split_fn = sroot / ("%s.txt" % ds_split)
    split_names = get_vid_names(split_fn)

    # -- read path names for split --
    names = []
    for dname in iroot.iterdir():
        name = str(dname.stem)
        base_name = name.split("_")[0]
        if not(base_name in split_names): continue
        tframes = len(list((iroot / name).iterdir()))
        nsubs = tframes - nframes + 1
        assert nsubs > 0
        for sub in range(nsubs):
            name_s = "%s+%02d" % (name,sub)
            names.append(name_s)
    return names

def read_data(name_s,iroot,nframes,bw=False):

    # -- split name --
    name,fstart = name_s.split("+")
    fstart = int(fstart)

    # -- load clean --
    paths = sorted(list((iroot / name).iterdir()))
    paths = [paths[ti] for ti in range(fstart,fstart+nframes)]
    clean = read_video(paths,bw=bw)

    # -- get frame nums --
    fnums = []
    for path in paths:
        fnum = int(str(path.stem))
        fnums.append(fnum)
    fnums = th.from_numpy(np.array(fnums))

    return clean,fnums


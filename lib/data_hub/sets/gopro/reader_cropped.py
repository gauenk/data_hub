
import torch as th
import numpy as np
from PIL import Image
from pathlib import Path
from easydict import EasyDict as edict
from einops import rearrange,repeat

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

def read_names(iroot,nframes,ext="png"):
    """
    Just read the folder names
    """

    root = iroot / "cropped/train/input"
    names = []
    for dname in root.iterdir():
        name = str(dname.stem)
        tframes = len(paths_at_name(iroot,"input",name))
        nsubs = tframes - nframes + 1
        assert nsubs > 0
        for sub in range(nsubs):
            name_s = "%s+%02d" % (name,sub)
            names.append(name_s)
    return names

def paths_at_name(iroot,itype,name):
    root = iroot / "cropped/train/" / itype / name
    paths = sorted(list(root.iterdir()))
    return paths

def read_data(name_s,iroot,nframes):

    # -- split name --
    name,fstart = name_s.split("+")
    fstart = int(fstart)

    # -- load blur --
    paths = paths_at_name(iroot,"input",name)
    paths = [paths[ti] for ti in range(fstart,fstart+nframes)]
    blur = read_video(paths,bw=False)

    # -- load sharp --
    paths = paths_at_name(iroot,"groundtruth",name)
    paths = [paths[ti] for ti in range(fstart,fstart+nframes)]
    sharp = read_video(paths,bw=False)

    # -- get frame nums --
    fnums = []
    for path in paths:
        fnum = int(str(path.stem))
        fnums.append(fnum)
    fnums = th.from_numpy(np.array(fnums))

    return blur,sharp,fnums


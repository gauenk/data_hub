
import torch as th
import numpy as np
from PIL import Image
from pathlib import Path
from easydict import EasyDict as edict
from einops import rearrange,repeat
from .paths import FLOW_BASE # why not other paths? I think we can do it when time

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
    iroot = iroot / "train"

    # -- read path names for split --
    names = []
    for dname in iroot.iterdir():
        name = str(dname.stem)
        base_name = name.split("_")[0]
        if not(base_name in split_names): continue
        base_dir = iroot / name
        tframes = len(list(p for p in base_dir.iterdir() if p.suffix in [".png"]))
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
    base = iroot / "train" / name
    tmp = list(base.iterdir())
    crop_info = [p.stem for p in base.iterdir() if "crop" in p.stem][0]
    paths = sorted(list(p for p in base.iterdir() if p.suffix in [".png",".jpeg"]))
    # print(fstart,nframes,len(paths))
    paths = [paths[ti] for ti in range(fstart,fstart+nframes)]
    clean = read_video(paths,bw=bw)

    # -- get frame nums --
    fnums = []
    for path in paths:
        fnum = int(str(path.stem))
        fnums.append(fnum)
    fnums = th.from_numpy(np.array(fnums))

    # -- top-left corner --
    fstart,fend,top,left = crop_info.split("_")[1:]
    fstart,fend = fnums[0].item(),fnums[-1].item()
    loc = [fstart,fend,top,left]
    loc = [int(x) for x in loc]

    return clean,fnums,loc


# -=-=-=-=-=-=-=-=-=-=-=-
#
#      Read Flows
#
# -=-=-=-=-=-=-=-=-=-=-=-

def read_flows(read_bool,vid_name,noise_info,seed,loc,isize):

    # -- no read --
    if not(read_bool):
        return th.FloatTensor([]),th.FloatTensor([])
    og_vid_name = vid_name
    vid_name = "_".join(vid_name.split("+")[0].split("_")[:-2])

    # -- read --
    fflow,bflow = read_flow_mmap(vid_name,noise_info,seed)

    # -- region --
    t_start,t_end,h_start,w_start = loc
    h_size,w_size = isize[0],isize[1]
    h_end,w_end = h_start+h_size,w_start+w_size

    # -- crop --
    og_fflow_shape = fflow.shape
    fflow = fflow[t_start:t_end+1,:,h_start:h_end,w_start:w_end]
    bflow = bflow[t_start:t_end+1,:,h_start:h_end,w_start:w_end]

    # -- to torch --
    fflow = th.from_numpy(fflow.copy()).type(th.float32)
    bflow = th.from_numpy(bflow.copy()).type(th.float32)
    # print(loc,og_vid_name,og_fflow_shape,fflow.shape)

    # -- temporal edges --
    fflow[-1] = 0
    bflow[0] = 0

    return fflow,bflow

def read_flow_mmap(vid_name,noise_info,seed):
    # -- read flow --
    file_stem = read_flow_base(noise_info,seed)
    fflow_fn = FLOW_BASE / vid_name / ("%s_fflow.npy" % file_stem)
    bflow_fn = FLOW_BASE / vid_name / ("%s_bflow.npy" % file_stem)
    fflow = np.load(fflow_fn,mmap_mode="r")
    bflow = np.load(bflow_fn,mmap_mode="r")
    return fflow,bflow

def read_flow_base(noise_info,seed):
    ntype = noise_info.ntype
    if ntype == "g":
        return "g-%d_seed-%d" % (noise_info.sigma,seed)
    elif ntype == "pg":
        return "pg-%d-%d_seed-%d" % (noise_info.sigma,noise_info.rate,seed)
    else:
        raise ValueError("Uknown noise type to reading pre-computed optical flow.")



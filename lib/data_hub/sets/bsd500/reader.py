
import torch as th
import numpy as np
from PIL import Image
from pathlib import Path
from einops import rearrange,repeat
from .paths import FLOW_BASE # why not other paths? I think we can do it when time

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

def read_annos(paths):
    vid = []
    for path_t in paths:
        path_t = str(path_t).replace("JPEGImages","Annotations")
        path_t = path_t.replace("jpg","png")
        path_t = Path(path_t)
        if not path_t.exists(): break
        vid_t = Image.open(str(path_t))
        vid_t = np.array(vid_t)[None,]
        vid_t = vid_t.astype(np.float32)
        vid.append(vid_t)
    vid = np.stack(vid).astype(np.float32)
    return vid

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
    MAXF = 10000
    paths,frame_nums = [],[]
    for t in range(MAXF):
        vid_t = vid_dir / ("%05d.%s" % (t,ext))
        if not vid_t.exists(): break
        paths.append(vid_t)
        frame_nums.append(t)
    return paths,frame_nums

def get_vid_names(base,split):
    return sorted(list((base / split).iterdir()))

# def read_flows(read_bool,vid_name,noise_info,seed):
#     if not(read_bool):
#         return th.FloatTensor([]),th.FloatTensor([])
#     file_stem = read_flow_base(noise_info,seed)
#     path = FLOW_BASE / vid_name / file_stem
#     flows = np.load(path)
#     fflow = th.Tensor(flows['fflow']).type(th.float)
#     bflow = th.Tensor(flows['bflow']).type(th.float)
#     return fflow,bflow

# def read_flow_base(noise_info,seed):
#     ntype = noise_info.ntype
#     if ntype == "g":
#         return "g-%d_seed-%d.npz" % (noise_info.sigma,seed)
#     elif ntype == "pg":
#         return "pg-%d-%d_seed-%d.npz" % (noise_info.sigma,noise_info.rate,seed)
#     else:
#         raise ValueError("Uknown noise type to reading pre-computed optical flow.")

# def read_files(iroot,split,nframes,stride,ext="png"):
#     """

#     what is stride?

#     stride = 1
#     _ _ _ _ _ _ _
#     x x x
#       x x x
#         x x x

#     stride = 2
#     _ _ _ _ _ _ _
#     x x x
#         x x x
#             x x x

#     what is dilation?

#     dilation = 2
#     _ _ _ _ _ _ _
#     x   x   x
#         x   x   x

#     """

#     # -- not input yet --
#     dil = 1

#     # -- get vid names in set --
#     img_names = get_img_names(iroot,split)

#     return img_names

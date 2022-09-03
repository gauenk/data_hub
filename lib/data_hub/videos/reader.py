
import json,rawpy
import numpy as np
from PIL import Image
from pathlib import Path
from einops import rearrange
from easydict import EasyDict as edict

def optional(pydict,key,default):
    if pydict is None: return default
    if not(key in pydict): return default
    return pydict[key]

def get_menu():
    db = "/home/gauenk/Documents/packages/data_hub/lib/data_hub/videos/paths.txt"
    with open(db,"r") as f:
        data = json.load(f)
    return data

def get_vid_sets(root):
    root = Path(root)
    vid_sets = {f.stem:f for f in root.iterdir()}
    return vid_sets

def read_video(path,nframes,fmt,ext,itype):
    if itype == "rgb":
        return read_rgb_video(path,nframes,fmt,ext)
    elif itype == "raw":
        return read_raw_video(path,nframes,fmt)
    else:
        raise ValueError(f"Uknown video type [{itype}]")

def read_raw_video(path,nframes,fmt):
    if isinstance(path,str):
        path = Path(path)
    vid = []
    ext = list(path.iterdir())[0].suffix[1:]
    for t in range(nframes):
        path_t = path / ("%05d.%s" % (t,ext))
        if not path_t.exists(): break
        vid_t = rawpy.imread(str(path_t))
        vid.append(vid_t)
    return vid

def read_rgb_video(path,nframes,fmt,ext):
    if isinstance(path,str):
        path = Path(path)
    vid = []
    for t in range(nframes):
        path_t = path / (("%s.%s" % (fmt,ext)) % t)
        if not path_t.exists(): break
        vid_t = Image.open(str(path_t)).convert("RGB")
        vid_t = (np.array(vid_t)*1.).astype(np.float32)
        vid_t = rearrange(vid_t,'h w c -> c h w')
        vid.append(vid_t)
    vid = np.stack(vid)
    return vid

def get_video_cfg(vid_set,vid_name,nframes=None,frame_fmt=None):
    cfg = edict()
    cfg.vid_set = vid_set
    cfg.vid_name = vid_name
    if not(nframes is None): cfg.nframes = nframes
    if not(frame_fmt is None): cfg.frame_fmt = frame_fmt
    return cfg

def read_frame(path,itype):
    if itype == "rgb":
        img = Image.open(str(path)).convert("RGB")
        img = (np.array(img)*1.).astype(np.float32)
        img = rearrange(img,'h w c -> c h w')
        return img
    elif itype == "raw":
        img = rawpy.imread(str(path))
        return img
    else:
        raise ValueError(f"Uknown video type [{itype}]")

def load_frame(cfg):

    # -- get data --
    menu = get_menu()
    set_info = menu[cfg.vid_set]
    root = Path(set_info['root'])
    ext = list(root.iterdir())[0].suffix[1:]
    path = root / ("%s.%s" % (cfg.vid_name,ext))
    assert path.exists()

    # -- pick --
    fmt = optional(cfg,"frame_fmt","%05d")
    itype = set_info['itype']
    vid = read_frame(path,itype)

    return vid


def load_video(cfg):

    # -- get data --
    menu = get_menu()
    set_info = menu[cfg.vid_set]
    assert set_info['cat'] == "vid"

    # -- pick set --
    vid_sets = get_vid_sets(set_info['root'])
    set_path = vid_sets[cfg.vid_name]

    # -- pick --
    nframes = optional(cfg,"nframes",85)
    fmt = optional(cfg,"frame_fmt","%05d")
    ext = optional(cfg,"frame_ext","png")
    itype = set_info['itype']
    vid = read_video(set_path,nframes,fmt,ext,itype)

    return vid

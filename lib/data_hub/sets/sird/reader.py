

import cache_io
import numpy as np
import torch as th
from PIL import Image
from pathlib import Path
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

def part_list(path):
    xy_l = []
    for path_i in path.iterdir():    
        _,x,y = path_i.stem.split("_")
        x,y = int(x),int(y)
        xy_l.append([x,y])
    # -- numpify --
    xy_l = np.array(xy_l)

    # -- order by pair ["lex"] --
    order = np.lexsort(xy_l.T)
    xy_l = xy_l[order]

    return xy_l
        
def isize2xy(isize):
    if isize in [None,"none"]: return -1,-1
    nx = (isize[0]-1)//256+1
    ny = (isize[1]-1)//256+1
    num = max(nx,ny)
    return num

def get_subgrid(grid,num):
    if num == -1: return grid
    assert num > 0
    order = np.random.permutation(len(grid))[:num]
    return [grid[o] for o in order]

def read_parts(path,isize,ext="png"):
    name = path.stem#.split("-")[0]
    xy_l = part_list(path)
    num = isize2xy(isize)
    xy_l = get_subgrid(xy_l,num)
    frame = []
    for xy in xy_l:
        path_xy = "%s_%d_%d.%s" % (name,x,y,ext)
        path_xy = path / path_xy
        frame_xy = Image.open(str(path_xy))
        frame_x.append(frame_xy)
    frame_x = np.concatenate(frame_x,0)
    frame.append(frame_x)
    frame = np.concatenate(frame,1)
    print(frame.shape)
    exit(0)
    return vid

def get_path_xy(path_t_dir,xy,ext="png"):
    x,y = xy
    name = path_t_dir.stem
    path_xy = "%s_%d_%d.%s" % (name,x,y,ext)
    return path_t_dir / path_xy

def read_video(paths,xy,bw=False,isize=None,ext="png"):

    """
    [paths]: the paths to the frames with cropped regions as files in dir
    [bw]: convert to black&white?
    [isize]: Randomly grab a spatial region the size of "isize"
    """
    # -- pair --
    vid = []
    for path_t_dir in paths:
        if not path_t_dir.exists(): break
        path_t = get_path_xy(path_t_dir,xy,ext)
        vid_t = Image.open(str(path_t))
        if bw: vid_t = np.array(vid_t.convert("L"))[...,None]
        else: vid_t = np.array(vid_t.convert("RGB"))
        vid_t = vid_t.astype(np.float32)
        vid_t = rearrange(vid_t,'h w c -> c h w')
        vid.append(vid_t)
    vid = np.stack(vid).astype(np.float32)

    # -- clean --
    name = paths[0].parents[0].stem
    x,y = xy
    stem = "%s_%d_%d.%s" % (name,x,y,ext)
    path_c = str(paths[0].parents[0]).replace("rainy","gt")
    path_c = Path(path_c) / stem
    clean = Image.open(str(path_c))
    clean = np.array(clean).astype(np.float32)
    clean = rearrange(clean,'h w c -> 1 c h w')

    # -- torch --
    vid = th.from_numpy(vid)
    clean = th.from_numpy(clean)

    return vid,clean

def frame_name_from_dir(vid_dir,t):
    vid_stem = vid_dir.stem
    frame_name = vid_stem + ("-%s" % t)
    return frame_name

def get_frame_ids(vid_dir):
    fids = []
    for fdir in vid_dir.iterdir():
        fids.append(int(fdir.stem.split("-")[-1]))
    fids = sorted(fids) # deterministic
    return fids

def get_vid_paths(vid_dir):
    fids = get_frame_ids(vid_dir)
    paths,frame_nums = [],[]
    for t in fids:
        name = frame_name_from_dir(vid_dir,t)
        vid_t = vid_dir / name
        if not vid_t.exists(): break
        paths.append(vid_t)
        frame_nums.append(t)
    return paths,frame_nums

def get_split_names(vid_fn):
    with open(vid_fn,"r") as f:
        names = f.readlines()
    names = [name.strip() for name in names]
    return names

def expand_split_names(iroot,subdir,split_names):
    vid_paths = []
    path = iroot / subdir / "rainy"
    for name in split_names:

        # -- enumerate --
        path_n = path / name
        paths_n_raw = list(path_n.iterdir())

        # -- order [deterministic] --
        nums = []
        for paths_i in paths_n_raw:
            num = paths_i.stem.split("-")[1]
            nums.append(int(num))
        order = np.argsort(nums)

        # -- final --
        paths_n = [paths_n_raw[i] for i in order]
        vid_paths.extend(paths_n)
    return vid_paths

def read_files(iroot,sroot,ds_split,nframes,stride):
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

    # -- read cache --
    cache_dir = ".cache_io"
    cache_name = "sird"
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()
    exp = {"split":ds_split,"nframes":nframes,"stride":stride}
    uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
    result = cache.load_exp(exp)
    if not(result is None):
        return result['files']

    # -- not input yet --
    dil = 1
    subdir = "train" if ds_split in ["train","val"] else "test"

    # -- get vid names in set --
    split_fn = sroot / ("%s.txt" % ds_split)
    vid_names = get_split_names(split_fn)

    # -- get files --
    files = {'images':{},"fnums":{},'xy':{}}
    for vid_name in vid_names:
        vid_dir = iroot/subdir/"rainy"/vid_name
        vid_paths,frame_nums = get_vid_paths(vid_dir)
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

        # -- sub videos --
        for group in range(n_subvids):

            start_t = group * stride
            if n_subvids == 1: subvid_id = vid_name
            else: subvid_id = "%s_%d" % (vid_name,start_t)
            end_t = start_t + nframes_vid
            paths_t = [vid_paths[bnd(t,total_nframes)] for t in range(start_t,end_t)]

            # -- spatial regions --
            xy_list = part_list(paths_t[0]) # this doesn't work for nframes > 1
            for (x,y) in xy_list:

                # -- video id --
                vid_id = "%s_%d_%d" % (subvid_id,x,y)

                # -- same paths to videos --
                files['images'][vid_id] = paths_t

                # -- spatial id --
                files['xy'][vid_id] = [x,y]

                # -- extra --
                fnums_t = [frame_nums[bnd(t,total_nframes)] for t in range(start_t,end_t)]
                files['fnums'][vid_id] = fnums_t

    # -- save cache --
    results = {"files":files}
    cache.save_exp(uuid,exp,results)


    return files

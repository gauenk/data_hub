
import torch as th
import numpy as np
from PIL import Image
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

def read_video(vid_name,fnums,root,bw=False):
    vid = []
    for fid in fnums:
        path_t = root / "sequences" / vid_name.replace("-","/") / ("im%d.png" % (fid+1))
        assert path_t.exists()
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

def get_vid_names(vid_fn):
    with open(vid_fn,"r") as f:
        names = f.readlines()
    names = [name.strip().replace("/","-") for name in names]
    return names

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

def read_files(iroot,sroot,ds_split,nframes,stride,ext="png"):

    # -- get vid names in set --
    split_fn = sroot / ("%s.txt" % ds_split)
    vid_names = get_vid_names(split_fn)

    # -- get files --
    files = {'images':{},"fnums":{}}

    # -- all have 7 --
    dil = 1
    frame_nums = np.arange(7)
    total_nframes = 7
    assert nframes <= total_nframes

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

    # -- create subvids --
    fnums = []
    for group in range(n_subvids):
        start_t = group * stride
        end_t = start_t + nframes_vid
        fnums_t = [frame_nums[bnd(t,total_nframes)] for t in range(start_t,end_t)]
        fnums.append(fnums_t)

    # -- repeat frames --
    nvids = len(vid_names)
    fnums = np.array(fnums)
    fnums = np.repeat(fnums[None,:],nvids,0).reshape((nvids*n_subvids,-1))

    # -- repeat vids --
    vid_names = np.array(vid_names)
    vid_names = np.repeat(vid_names,n_subvids,0)

    assert len(fnums) == len(vid_names)

    files = {"images":vid_names,"fnums":fnums}
    # for vid_name in vid_names:
    #     for group in range(n_subvids):
    #         start_t = group * stride
    #         if n_subvids == 1: vid_id = vid_name
    #         else: vid_id = "%s:%d" % (vid_name,start_t)
    #         fnums_t = [frame_nums[bnd(t,total_nframes)] for t in range(start_t,end_t)]
    #         files['images'].append(vid_id)
    #         files['fnums'].append(fnums_t)

    return files

def read_files_old(iroot,sroot,ds_split,nframes,stride,ext="png"):
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
    split_fn = sroot / ("%s.txt" % ds_split)
    vid_names = get_vid_names(split_fn)

    # -- get files --
    files = {'images':{},"fnums":{}}
    for vid_name in vid_names:
        vid_dir = iroot/"sequences"/vid_name.replace("-","/")
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
            if n_subvids == 1: vid_id = vid_name
            else: vid_id = "%s:%d" % (vid_name,start_t)
            end_t = start_t + nframes_vid
            paths_t = [vid_paths[bnd(t,total_nframes)] for t in range(start_t,end_t)]
            files['images'][vid_id] = paths_t

            # -- extra --
            fnums_t = [frame_nums[bnd(t,total_nframes)] for t in range(start_t,end_t)]
            files['fnums'][vid_id] = fnums_t

    return files

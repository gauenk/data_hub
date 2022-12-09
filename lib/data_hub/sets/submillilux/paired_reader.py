
import torch as th
import numpy as np
from PIL import Image
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

def read_video(paths,bw=False):
    vid = []
    for path_t in paths:
        if not path_t.exists(): break
        if bw:
            vid_t = Image.open(str(path_t)).convert("L")
            vid_t = rearrange(np.array(vid_t),'h w -> 1 h w')
        else:
            vid_t = Image.open(str(path_t)).convert("RGB")
            vid_t = rearrange(np.array(vid_t),'h w c -> c h w')
        vid_t = (vid_t*1.).astype(np.float32)
        vid.append(vid_t)
    vid = np.stack(vid).astype(np.float32)
    vid = th.from_numpy(vid)
    return vid

def get_vid_names(vid_fn):
    with open(vid_fn,"r") as f:
        names = f.readlines()
    _names = [name.strip() for name in names]
    names = []
    for name in _names:
        if len(name) == 0: continue
        names.append(name)
    return names

def get_video_paths(vid_dir):
    MAXF = 10000
    noisy_paths = []
    gt_paths = []
    for t in range(MAXF):
        noisy_t = vid_dir / ("noisy_%05d.png" % t)
        gt_t = vid_dir / ("gt_%05d.png" % t)
        if not noisy_t.exists(): break
        if not gt_t.exists(): break
        noisy_paths.append(noisy_t)
        gt_paths.append(gt_t)
    return noisy_paths,gt_paths

def read_files(iroot,sroot,ds_split,nframes,dil=1,stride=1):

    # -- get vid names in set --
    split_fn = sroot / ("%s.txt" % ds_split)
    vid_names = get_vid_names(split_fn)

    # -- get files --
    files = {'images':{},"fnums":{}}
    for vid_name in vid_names:
        vid_dir = iroot/vid_name
        noisy_paths,gt_paths = get_video_paths(vid_dir)
        total_nframes = len(noisy_paths)
        assert total_nframes > 0
        frame_nums = np.arange(len(noisy_paths))

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

        for start_t in range(n_subvids):
            vid_id = "%s_%d" % (vid_name,start_t)
            end_t = start_t + nframes
            noisy_t = [noisy_paths[t%total_nframes] for t in range(start_t,end_t)]
            gt_t = [gt_paths[t%total_nframes] for t in range(start_t,end_t)]
            files['images'][vid_id] = [noisy_t,gt_t]

            # -- extra --
            fnums_t = [frame_nums[bnd(t,total_nframes)] for t in range(start_t,end_t)]
            files['fnums'][vid_id] = fnums_t

    return files

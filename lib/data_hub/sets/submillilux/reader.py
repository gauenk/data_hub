
import numpy as np
from PIL import Image
import scipy.io
from einops import rearrange,repeat

def read_mat_in_dir(ipath,nframes):
    vid = []
    for t in range(nframes):
        path_t = path / ("%d.mat" % t)
        if not path_t.exists(): break
        loaded_t = scipy.io.loadmat(path_t)
        img_t = loaded_t['noisy_list'].astype('float32')/2**16
        vid.append(img_t)
    vid = np.stack(vid)
    return vid

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

def read_video(paths):
    vid = []
    for path_t in paths:
        if not path_t.exists(): break
        vid_t = Image.open(str(path_t)).convert("RGB")
        vid_t = (np.array(vid_t)*1.).astype(np.float32)
        vid_t = rearrange(vid_t,'h w c -> c h w')
        vid.append(vid_t)
    vid = np.stack(vid).astype(np.float32)
    return vid

def read_mats(paths):
    vid = []
    for path_t in paths:
        if not path_t.exists(): break
        loaded_t = scipy.io.loadmat(path_t)
        img_t = loaded_t['noisy_list'].astype('float32')/2**16
        img_t = rearrange(img_t,'h w c -> c h w')
        vid.append(img_t)
    vid = np.stack(vid).astype(np.float32)
    return vid

def get_vid_names(vid_fn):
    with open(vid_fn,"r") as f:
        names = f.readlines()
    names = [name.strip() for name in names]
    return names

def get_video_paths(vid_dir,ext="dng"):
    MAXF = 10000
    paths,frame_nums = [],[]
    for t in range(MAXF):
        vid_t = vid_dir / ("%d.%s" % (t,ext))
        if not vid_t.exists(): break
        paths.append(vid_t)
        frame_nums.append(t)
    return paths,frame_nums

def read_files(iroot,sroot,ds_split,nframes,fskip,ext="dng"):

    # -- get vid names in set --
    split_fn = sroot / ("%s.txt" % ds_split)
    vid_names = get_vid_names(split_fn)

    # -- get files --
    files = {'images':{},"fnums":{}}
    for vid_name in vid_names:
        vid_dir = iroot/vid_name
        vid_paths,frame_nums = get_video_paths(vid_dir,ext)
        total_nframes = len(vid_paths)
        assert total_nframes > 0

        # -- pick number of sub frames --
        nframes_vid = nframes
        if nframes_vid <= 0:
              nframes_vid = total_nframes

        # -- compute num subframes --
        n_subvids = max((total_nframes-1)//fskip+1,1)

        # -- reflect bound --
        def bnd(num,lim):
            if num >= lim: return 2*(lim-1)-num
            else: return num

        for group in range(n_subvids):
            start_t = group * fskip
            if n_subvids == 1: vid_id = vid_name
            else: vid_id = "%s_%d" % (vid_name,start_t)
            end_t = start_t + nframes_vid
            paths_t = [vid_paths[bnd(t,total_nframes)] for t in range(start_t,end_t)]
            files['images'][vid_id] = paths_t

            # -- extra --
            fnums_t = [frame_nums[bnd(t,total_nframes)] for t in range(start_t,end_t)]
            files['fnums'][vid_id] = fnums_t

    return files

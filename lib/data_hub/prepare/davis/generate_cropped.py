from glob import glob
from tqdm import tqdm
import numpy as np
import os
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse
from pathlib import Path
np.random.seed(0)


def load_dir_fns(root):
    fns = []
    fids = []
    for fn in root.iterdir():
        fns.append(str(fn))
        fid = int(fn.stem)
        fids.append(fid)
    fids = sorted(fids)
    return fns,fids

def get_vid_from_fids(root,fids):
    imgs = []
    for fid in fids:
        img_f = root / ("%05d.jpg" % fid)
        imgs.append(str(img_f))
    return imgs

def load_vid_paths(root):
    root = Path(root)
    paths = {}
    skip_dirs = [".cache"]
    for path in root.iterdir():
        name = path.stem
        if name in skip_dirs: continue
        _,fids = load_dir_fns(path)
        paths[name] = {}
        paths[name]['root'] = path
        paths[name]['fids'] = fids
    return paths

def load_vid(fns):
    vid = []
    for fn in fns:
        img = cv2.imread(fn)
        vid.append(img)
    vid = np.stack(vid)
    return vid

def write_vid(vid,loc,fids,root,name_i,j,k):

    # -- create directory --
    path = os.path.join(root, '{}_{}_{}/'.format(name_i,j+1,k+1))
    os.makedirs(path)

    # -- create textfile with t1..t2 and top-left location --
    tmin,tmax = min(fids),max(fids)
    np.save(Path(path) / ("crop_%d_%d_%d_%d.npy" % (tmin,tmax,loc[0],loc[1])),loc)

    # -- save each frame --
    t = vid.shape[0]
    for ti in range(t):
        fid_ti = fids[ti]
        path_t = os.path.join(path,'%05d.png' % fid_ti)
        vid_t = vid[ti]
        cv2.imwrite(path_t, vid_t)

def save_files(i,name_i,tar,root,fids,NF,PS,NUM_PATCHES):

    # -- num sub vids --
    T = len(fids)
    NUM_SVIDS = T-NF+1

    for j in range(NUM_SVIDS):

        # -- randomly select frames --
        T = len(fids)
        fids_s = [fids[ti] for ti in range(j,NF+j)]

        # -- load burst --
        img_fns = get_vid_from_fids(root,fids_s)
        clean_vid = load_vid(img_fns)

        # -- unpack shapes --
        H = clean_vid.shape[1]
        W = clean_vid.shape[2]

        # -- iterate across patches --
        for k in range(NUM_PATCHES):

             # -- spatial crop --
             rr = np.random.randint(0, H - PS)
             cc = np.random.randint(0, W - PS)
             clean_patch = clean_vid[:, rr:rr + PS, cc:cc + PS, :]
             assert PS == clean_patch.shape[-3]
             assert PS == clean_patch.shape[-2]
             loc = [rr,cc]

             # -- write --
             write_vid(clean_patch,loc,fids_s,tar,name_i,j,k)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate patches from Full Resolution images')
    parser.add_argument('--src_dir', default='./JPEGImages/480p', type=str,
                        help='Directory for full resolution images')
    parser.add_argument('--tar_dir', default='./cropped/train/',type=str,
                        help='Directory for image patches')
    parser.add_argument('--nf', default=10, type=int, help='Num of Image Frames')
    parser.add_argument('--ps', default=256, type=int, help='Image Patch Size')
    parser.add_argument('--num_patches', default=3, type=int,
                        help='Number of patches per image')
    parser.add_argument('--num_cores', default=10, type=int,
                        help='Number of CPU Cores')
    args = parser.parse_known_args()[0]
    return args

def run(base):

    # -- unpack --
    args = parse_args()
    src = Path(base) / args.src_dir
    tar = Path(base) / args.tar_dir
    NF = args.nf
    PS = args.ps
    NUM_PATCHES = args.num_patches
    NUM_CORES = args.num_cores
    paths = load_vid_paths(src)
    names = list(paths.keys())
    names = sorted(names)

    if os.path.exists(tar):
        os.system("rm -r {}".format(tar))
    os.makedirs(tar)

    Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i,names[i],tar,
                                                   paths[names[i]]['root'],
                                                   paths[names[i]]['fids'],
                                                   NF,PS,NUM_PATCHES)
                               for i in tqdm(range(len(names))))

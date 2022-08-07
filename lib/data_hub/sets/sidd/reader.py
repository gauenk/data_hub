
import os

import numpy as np
import torch as th
from einops import rearrange

from PIL import Image
import pdb,hdf5storage
from pathlib import Path

def mode_info(mode):
    if mode == "rgb":
        return "SRGB","PNG"
    elif mode == "raw":
        return "RAW","MAT"
    else:
        raise ValueError(f"Uknown mode [{mode}]")

def read_file_pair(fn_pair,mode):
    pair = []
    for fn in fn_pair:
        if mode == "rgb":
            img_i = Image.open(fn)
            img_i = img_i.convert("RGB")
        elif mode == "raw":
            img_i = hdf5storage.loadmat(str(fn))['x']
            img_i = img_i[...,None]
        img_i = np.array(img_i)[None,:]
        img_i = rearrange(img_i,'g h w c -> g c h w')
        img_i = img_i.astype(np.float32)
        img_i = th.from_numpy(img_i).contiguous()
        pair.append(img_i)
    return pair[0],pair[1]

def read_cropped_medium_files(iroot,mode):
    iroot = Path(iroot)
    in_dir = iroot / "input"
    gt_dir = iroot / "groundtruth"
    pairs,fids = [],[]
    for fn in in_dir.iterdir():
        fid,fnum = str(fn.stem).split("_")
        fid,fnum = int(fid),int(fnum.split(".")[0])
        in_fn = in_dir / ("%s_%s.png" % (fid,fnum))
        gt_fn = gt_dir / ("%s_%s.png" % (fid,fnum))
        pairs.append([in_fn,gt_fn])
        fids.append(fid)
    order = np.argsort(fids)
    pairs  = [pairs[o] for o in order]
    nimgs = len(pairs)
    groups = ["%02d" % gid for gid in range(nimgs)]
    return pairs,groups

def read_medium_files(iroot,mode):
    iroot = Path(iroot)
    istr,suf = mode_info(mode)
    fns,fids = [],[]
    for fn in iroot.iterdir():
        if not(fn.is_dir()): continue
        str_id = str(fn.stem).split("_")[0]
        fid = int(str_id)
        for num in [10,11]:
            fid_num = "%s_%2d" %(fid,num)
            pair_fns = []
            for iver in ["GT","NOISY"]:
                fn_i = fn / ("%s_%s_%s_%03d.%s" % (str_id,iver,istr,num,suf))
                if mode == "raw": # "MAT" to "mat" [silly, yes]
                    fn_o = fn_i
                    fn_i = fn / ("%s_%s_%s_%03d.mat" % (str_id,iver,istr,num))
                    if not(fn_i.exists()): os.symlink(fn_o,fn_i)
                assert fn_i.exists()
                pair_fns.append(fn_i)
            fns.append(pair_fns)
            fids.append(fid_num)

    order = np.argsort(fids)
    fns = [fns[o] for o in order]
    nimgs = len(fns)
    groups = ["%02d" % gid for gid in range(nimgs)]
    return fns,groups

def read_files(iroot,mode):
    iroot = Path(iroot)
    istr,suf = mode_info(mode)
    fns,fids = [],[]
    for fn in iroot.iterdir():
        if not(fn.is_dir()): continue
        str_id = str(fn.stem).split("_")[0]
        fid = int(str_id)
        fn_i = fn / ("%s_NOISY_%s_010.%s" % (str_id,istr,suf))
        if mode == "raw":
            fn_o = fn_i
            fn_i = fn / ("%s_NOISY_%s_010.mat" % (str_id,istr))
            if not(fn_i.exists()): os.symlink(fn_o,fn_i)
        assert fn_i.exists()
        fns.append(fn_i)
        fids.append(fid)

    order = np.argsort(fids)
    fns = [fns[o] for o in order]
    nimgs = len(fns)
    groups = ["%02d" % gid for gid in range(nimgs)]
    return fns,groups

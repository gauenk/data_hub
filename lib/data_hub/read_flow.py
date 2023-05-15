
import torch as th
import numpy as np
import numpy.random as npr
from PIL import Image
from pathlib import Path
from easydict import EasyDict as edict
from einops import rearrange,repeat

# -=-=-=-=-=-=-=-=-=-=-=-
#
#      Read Flows
#
# -=-=-=-=-=-=-=-=-=-=-=-

def read_flows(FLOW_BASE,read_bool,vid_name,noise_info,seed,loc,isize):

    # -- no read --
    if not(read_bool):
        return th.FloatTensor([]),th.FloatTensor([])

    # -- read --
    fflow,bflow = read_flow_mmap(FLOW_BASE,vid_name,noise_info,seed)

    # -- region --
    t_start,t_end,h_start,w_start = loc
    h_size,w_size = isize[0],isize[1]
    h_end,w_end = h_start+h_size,w_start+w_size

    # -- crop --
    og_fflow_shape = fflow.shape
    fflow = fflow[t_start:t_end,:,h_start:h_end,w_start:w_end]
    bflow = bflow[t_start:t_end,:,h_start:h_end,w_start:w_end]

    # -- to torch --
    fflow = th.from_numpy(fflow.copy()).type(th.float32)
    bflow = th.from_numpy(bflow.copy()).type(th.float32)

    # -- temporal edges --
    fflow[-1] = 0
    bflow[0] = 0

    return fflow,bflow

def read_flow_mmap(FLOW_BASE,vid_name,noise_info,seed):
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
    elif ntype == "msg":
        sigma = npr.choice([15,30,50],size=1).item()
        return "g-%d_seed-%d" % (sigma,seed)
    elif ntype == "pg":
        return "pg-%d_seed-%d" % (noise_info.rate,seed)
        # return "pg-%d-%d_seed-%d" % (noise_info.sigma,noise_info.rate,seed)
    else:
        raise ValueError("Uknown noise type to reading pre-computed optical flow.")



"""

This file has functions for a user to crop a video
by randomly sampling a coordinate
with probability propto an edge

"""

# -- rand nums --
import random

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- image pair dataset --
import random
from PIL import Image
import torch.utils.data as data
import torch.nn.functional as nnf
import torchvision.transforms as transforms
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as tf

def crop_vid(vid,cropmode,isize,region_temp):
    if cropmode is None or cropmode == "none": return vid
    if cropmode == "rand":
        vid = run_rand_crop(vid,isize)
        rtn = vid
    elif cropmode in ["coords","coords_sobel","region","region_sobel"]:
        sobel_vid = apply_sobel_filter(vid)
        region = sample_sobel_region(sobel_vid,region_temp)
        region = th.IntTensor(region)
        rtn = region
    elif cropmode in ["sobel"]:
        sobel_vid = apply_sobel_filter(vid[0])
        region = sample_sobel_region(sobel_vid,region_temp)
        region = th.IntTensor(region)
        vid_cc = [rslice(vid[l],region) for l in range(len(vid))]
        rtn = vid_cc
    elif cropmode in ["coords_rand","region_rand"]:
        vshape = vid.shape
        region = sample_rand_region(vshape,region_temp)
        region = th.IntTensor(region)
        rtn = region
    elif cropmode in ["center_crop","center","centercrop"]:
        vid_cc = run_center_crop(vid,isize)
        rtn = vid_cc
    else:
        raise NotImplementedError(f"Uknown crop mode [{cropmode}]")
    return rtn

def run_center_crop(vid_l,isize):
    # -- ensure list --
    single = False
    if not isinstance(vid_l,list):
        single = True
        vid_l = [vid_l]

    # -- all center crop --
    crop_l = []
    for vid in vid_l:
        vid_c = tf.center_crop(vid,isize)
        crop_l.append(vid_c)

    # -- return single if single input --
    if single: crop_l = crop_l[0]
    return crop_l

def run_rand_crop(vid_l,isize):

    # -- ensure list --
    single = False
    if not isinstance(vid_l,list):
        single = True
        vid_l = [vid_l]

    # -- get crop info --
    rand_crop = RandomCrop(isize)
    i, j, h, w = RandomCrop.get_params(
        vid_l[0], output_size=isize)

    # -- crop all same --
    crop_l = []
    for vid in vid_l:
        vid_c = tf.crop(vid, i, j, h, w)
        crop_l.append(vid_c)

    # -- return single if single input --
    if single: crop_l = crop_l[0]
    return crop_l

def create_sobel_filter():
    # -- get sobel filter to detect rough spots --
    sobel = th.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_t = sobel.t()
    sobel = sobel.reshape(1,3,3)
    sobel_t = sobel_t.reshape(1,3,3)
    weights = th.stack([sobel,sobel_t],dim=0)
    return weights

def apply_sobel_filter(image):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    C = image.shape[-3]
    weights = create_sobel_filter()
    weights = weights.to(image.device)
    weights = repeat(weights,'b 1 h w -> b c h w',c=C)
    edges = nnf.conv2d(image,weights,padding=1,stride=1)
    edges = ( edges[:,0]**2 + edges[:,1]**2 ) ** (0.5)

    # -- compute spatial average (to find "good points") --
    weights = th.FloatTensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9.
    weights = weights[None,None,:,:].to(image.device)
    spatial_ave_edges = nnf.conv2d(edges[:,None,:,:],weights,padding=1,stride=1)
    return spatial_ave_edges

def point2range(p,reg,lb,ub):

    # -- special case --
    if reg in [-1,0]: reg = ub - lb

    # -- pmin --
    pmin = p-((reg-1)//2+1)
    pmin = max(pmin,0)

    # -- p max --
    pmax = pmin + reg
    pmax = min(pmax,ub)

    # -- shift left if needed --
    curr_reg = pmax - pmin # current size
    lshift = reg - curr_reg # remaining
    pmin = max(pmin - lshift,0)

    # -- assign --
    pstart = pmin
    pend = pmax

    # -- assert --
    info = "%d,%d,%d,%d,%d" % (pstart,lb,pend,ub,reg)
    assert (pstart >= lb) and (pend <= ub),info
    assert (pend - pstart) <= reg,info
    return pstart,pend

def rslice(vid,region):
    t0,t1,h0,w0,h1,w1 = region
    return vid[t0:t1,:,h0:h1,w0:w1]

def sample_sobel_point(sobel_vid):
    t,c,h,w = sobel_vid.shape
    sobel_vid = th.mean(sobel_vid,1)
    # print(sobel_vid.shape)
    # assert sobel_vid.shape[1] == 1,"only one color channel."
    hw = h * w
    size = t * h * w
    ind = int(th.multinomial(sobel_vid.ravel(),1).item())
    ti = ind // hw
    hi = (ind%hw)//w
    wi = (ind%hw)%w
    info = str(sobel_vid.shape) + (" %d,%d,%d,%d" % (ind,ti,hi,wi))
    assert ti < t,info
    assert hi < h,info
    assert wi < w,info
    point = [ti,hi,wi]
    return point

def sample_rand_point(vshape):
    t,c,h,w = vshape
    ti = int(np.random.rand(1).item()*t)
    hi = int(np.random.rand(1).item()*h)
    wi = int(np.random.rand(1).item()*w)
    assert ti < t
    assert hi < h
    assert wi < w
    point = [ti,hi,wi]
    return point

def point_to_region(point,reg_temp,nframes,height,width):
    # -- unpack template --
    rtemp = reg_temp.split("_")
    reg_nframes = int(rtemp[0])
    reg_height = int(rtemp[1])
    reg_width = int(rtemp[2])

    # -- center region --
    fstart,fend = point2range(point[0],reg_nframes,0,nframes)
    top,btm = point2range(point[1],reg_height,0,height)
    left,right = point2range(point[2],reg_width,0,width)

    # -- create region --
    region = [fstart,fend,top,left,btm,right]
    return region

def sample_sobel_region(sobel_vid,reg_temp):
    nframes,c,height,width = sobel_vid.shape
    point = sample_sobel_point(sobel_vid)
    region = point_to_region(point,reg_temp,nframes,height,width)
    return region

def sample_rand_region(vshape,reg_temp):
    t,c,h,w = vshape
    point = sample_rand_point(vshape)
    region = point_to_region(point,reg_temp,t,h,w)
    return region

def get_center_region(vshape,region_temp):
    t,c,h,w = vshape
    rt,rh,rw = region_temp.split("_")
    rt,rh,rw = int(rt),int(rh),int(rw)
    if rt == 0: rt = t

    ts = t//2 - rt//2
    hs = h//2 - rh//2
    ws = w//2 - rw//2

    te = ts + rt
    he = hs + rh
    we = ws + rw

    region = [ts,te,hs,ws,he,we]

    return region

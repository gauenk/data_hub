
import numpy as np
from torchvision.transforms import functional as tvf

def rescale_imgs(iscale,*vids):
    if iscale > 0.99 or iscale < 1.01: return vids
    s_vids = []
    for vid in vids:
        H,W = vid.shape[-2:]
        sH,sW = int(iscale*H),int(iscale*W)
        s_vids.append(tvf.rescale(vid,(sH,sW)))
    return s_vids

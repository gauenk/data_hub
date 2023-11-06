
import numpy as np
from torchvision.transforms import functional as tvf

def rescale_imgs(iscale,*vids):
    if np.isclose(iscale,1): return vids
    s_vids = []
    for vid in vids:
        if vid.ndim == 1:
            s_vids.append(vid)
            continue
        H,W = vid.shape[-2:]
        sH,sW = int(iscale*H),int(iscale*W)
        s_vids.append(tvf.resize(vid,(sH,sW)))
    return s_vids

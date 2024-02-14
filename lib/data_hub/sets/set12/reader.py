
try:
    import cv2
except:
    pass
import numpy as np
from PIL import Image
from einops import rearrange,repeat

def read_video(path):
    img = np.array(Image.open(str(path)))
    img = repeat(img,'h w -> h w c',c=1)
    img = img.astype(np.float32)
    img = rearrange(img,'h w c -> c h w')
    return img

def get_image_names(vid_fn):
    with open(vid_fn,"r") as f:
        names = f.readlines()
    names = [name.strip() for name in names]
    return names

def read_files(iroot,sroot,ds_split,ext="png"):

    # -- get vid names in set --
    split_fn = sroot / ("%s.txt" % ds_split)
    image_names = get_image_names(split_fn)

    # -- get files --
    files = {'images':{}}
    for image_name in image_names:
        image_fn = iroot/("%s.%s" % (image_name,ext))
        files['images'][image_name] = image_fn

    return files

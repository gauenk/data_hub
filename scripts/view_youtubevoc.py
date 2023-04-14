"""

View some of the CityScapes data

"""


import data_hub
from easydict import EasyDict as edict
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks
plt.rcParams["savefig.bbox"] = 'tight'

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig("output/view_youtube.png")
    plt.cla()
    plt.clf()
    plt.close()

def main():

    # -- load data --
    cfg = edict({
        "dname":"youtube_voc",
        "sigma":30,
        "nframes":5,
    })
    data,loaders = data_hub.sets.load(cfg)
    with_masks = []

    # -- info --
    for dset in data:
        print("Length = ",len(data[dset]))

    # -- loop --
    inds = [0,100,200]
    for i in inds:

        # -- unpack --
        sample = data.tr[i]
        vid = sample['clean']
        annos = sample['annos']

        # -- apply masks --
        vid = vid.type(th.uint8)
        uniqs = th.unique(annos)
        masks = th.stack([annos == u for u in uniqs],1)
        for j in range(min(len(vid),2)):
            with_masks.append(draw_segmentation_masks(vid[j],
                                                      masks=masks[j], alpha=0.7))
    show(with_masks)

if __name__ == "__main__":
    main()

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
# import pylab
# pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='/depot/chan129/data/COCO/'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)
print(coco)

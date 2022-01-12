import json
from pycocotools.coco import COCO
import os

import matplotlib.pyplot as plt


ann_dir = 'annotations/t1_from_t2/all.json'

img_folder = ''

coco = COCO(ann_dir)

ImgId = 12

img = coco.loadImgs([ImgId])[0]

plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'])
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
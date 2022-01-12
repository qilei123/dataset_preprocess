import json
from pycocotools.coco import COCO
import os
import skimage.io as io
import matplotlib.pyplot as plt


ann_dir = '/home/qilei/.TEMP/teeth_data2/20211213_t3/annotations/instances_default.json'

img_folder = '/home/qilei/.TEMP/teeth_data2/20211213_t3/images/'

coco = COCO(ann_dir)

ImgId = 12

img = coco.loadImgs([ImgId])[0]

img['file_name'] = img_folder+img['file_name']
I = io.imread(img['file_name'])
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'])
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
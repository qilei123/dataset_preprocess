<<<<<<< HEAD
import json
from pycocotools.coco import COCO
import os
import skimage.io as io
import matplotlib.pyplot as plt

dataset_dir = "/home/qilei/.TEMP/TEETH3/"

ann_dir = '/home/qilei/.TEMP/TEETH3/annotations/train_1_3_roi.json'

img_folder = '/home/qilei/.TEMP/TEETH3/images/'

coco = COCO(ann_dir)

ImgId = 8

img = coco.loadImgs([ImgId])[0]

if 'Wide' in img['file_name']:
    img['file_name'] = img['file_name'].replace('andover','andover_wide')

img['file_name'] = os.path.join(img_folder,img['file_name'])
I = io.imread(img['file_name'])


plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'])
anns = coco.loadAnns(annIds)
coco.showAnns(anns,draw_bbox=True)
plt.axis('off')
plt.imshow(I)

plt.show()
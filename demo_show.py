import json
from pycocotools.coco import COCO
import os
import skimage.io as io
import matplotlib.pyplot as plt

dataset_dir = "/home/qilei/.TEMP/WJ/"

ann_dir = '/home/qilei/.TEMP/WJ/annotations/train.json'

img_folder = '/home/qilei/.TEMP/WJ/images/'

coco = COCO(ann_dir)

ImgId = 1538

img = coco.loadImgs([ImgId])[0]

if 'Wide' in img['file_name']:
    img['file_name'] = img['file_name'].replace('andover','andover_wide')

img['file_name'] = os.path.join(img_folder,img['file_name'])
I = io.imread(img['file_name'])


plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'])
anns = coco.loadAnns(annIds)
print(anns)
coco.showAnns(anns,draw_bbox=True)
plt.axis('off')
plt.imshow(I)

plt.show()
import json
from pycocotools.coco import COCO
import os
import skimage.io as io
import matplotlib.pyplot as plt


ann_dir = '/data3/qilei_chen/DATA/trans_drone/annotations/test_AW_obb.json'

img_folder = '/data3/qilei_chen/DATA/trans_drone/images/'

coco = COCO(ann_dir)

ImgId = 12

img = coco.loadImgs([ImgId])[0]

if 'Wide' in img['file_name']:
    img['file_name'] = img['file_name'].replace('andover','andover_wide')

img['file_name'] = os.path.join(img_folder,img['file_name'])
I = io.imread(img['file_name'])


plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'])
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.axis('off')
plt.imshow(I)
plt.show()
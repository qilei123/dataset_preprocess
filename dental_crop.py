from pycocotools.coco import COCO
import json
import cv2
import os

def dental_crop_with_max_bounds(anno_dir,imgs_dir,croped_imgs_dir):
    
    temp_annotation = json.load(open(anno_dir))
    coco = COCO(anno_dir)
    temp_annotation["images"] = []
    temp_annotation["annotations"] = []

    for ImgId in coco.getImgIds():
        img = coco.loadImgs([ImgId])[0]

        annIds =  coco.getAnnIds(ImgId)
        anns = coco.loadAnns(annIds)
        
        # Get the boundary box that can contain all the object instances on the image

        bound_x1 = float('inf')
        bound_y1 = float('inf')
        bound_x2 = 0
        bound_y2 = 0 

        for ann in anns:
            bound_x1 = bound_x1 if bound_x1<ann['bbox'][0] else int(ann['bbox'][0])

            bound_y1 = bound_y1 if bound_y1<ann['bbox'][1] else ann['bbox'][1]
            
            bound_x2 = bound_x2 if bound_x2>ann['bbox'][0]+ann['bbox'][2] else ann['bbox'][0]+ann['bbox'][2]
            
            bound_y2 = bound_y2 if bound_y2>ann['bbox'][1]+ann['bbox'][3] else ann['bbox'][1]+ann['bbox'][3]            

        temp_annotation["images"].append(img)
        image = cv2.imread(os.path.join(imgs_dir,img['file_name']))
        
        if len(anns)>0:

            bound_box = [int(bound_x1),int(bound_y1),int(bound_x2),int(bound_y2)]
        
        else:

            bound_box = [0, 0, image.shape[1], image.shape[0]]


        cropped_image = image[bound_box[1]:bound_box[3],bound_box[0]:bound_box[2]]
        
        for ann in anns:
            ann['bbox'][0]-=bound_box[0]
            ann['bbox'][1]-=bound_box[1]

            for idx,xy in enumerate(ann['segmentation'][0]):
                ann['segmentation'][0][idx] = xy - bound_box[idx%2]
            
            temp_annotation['annotations'].append(ann)


        img_dir = os.path.join(croped_imgs_dir,img['file_name'])
        os.makedirs(img_dir.replace(os.path.basename(img_dir),''),exist_ok=True)
        cv2.imwrite(os.path.join(croped_imgs_dir,img['file_name']),cropped_image)            

    with open(anno_dir.replace('.json','_crop.json'), 'w') as outfile:
        json.dump(temp_annotation, outfile)    

if __name__=="__main__":
    anno_dir = "/home/qilei/.TEMP/TEETH3/annotations/train_1_3.json"
    imgs_dir = "/home/qilei/.TEMP/TEETH3/images/"
    croped_imgs_dir = "/home/qilei/.TEMP/TEETH3/images_crop1/"
    dental_crop_with_max_bounds(anno_dir,imgs_dir,croped_imgs_dir)
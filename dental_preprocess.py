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

        
        image = cv2.imread(os.path.join(imgs_dir,img['file_name']))
        
        if len(anns)>0:

            bound_box = [int(bound_x1),int(bound_y1),int(bound_x2),int(bound_y2)]
        
        else:

            bound_box = [0, 0, image.shape[1], image.shape[0]]


        cropped_image = image[bound_box[1]:bound_box[3],bound_box[0]:bound_box[2]]
        
        # set the size of new cropped images in the annotation with integer
        img['width'] = int(bound_box[2]-bound_box[0])
        img['height'] = int(bound_box[3]-bound_box[1])

        temp_annotation["images"].append(img)

        for ann in anns:
            # fix the bounding box position according to the cropped coordination
            ann['bbox'][0] = round(ann['bbox'][0]-bound_box[0],2)
            ann['bbox'][1] = round(ann['bbox'][1]-bound_box[1],2)

            # fix the polygon coordination
            for idx,xy in enumerate(ann['segmentation'][0]):
                ann['segmentation'][0][idx] = round(xy - bound_box[idx%2],2)
            
            temp_annotation['annotations'].append(ann)


        img_dir = os.path.join(croped_imgs_dir,img['file_name'])
        os.makedirs(img_dir.replace(os.path.basename(img_dir),''),exist_ok=True)
        cv2.imwrite(os.path.join(croped_imgs_dir,img['file_name']),cropped_image)            

    with open(anno_dir.replace('.json','_crop.json'), 'w') as outfile:
        json.dump(temp_annotation, outfile)    

def redefine_id(ids_map,cat_names,anno_dir):

    temp_annotation = json.load(open(anno_dir))
    coco = COCO(anno_dir)
    temp_annotation["annotations"] = []    

    temp_annotation['categories']=[]

    for id in cat_names:
        new_cats = {"id": id, "name": cat_names[id], "supercategory": ""}
        temp_annotation['categories'].append(new_cats)

    for ImgId in coco.getImgIds():
        img = coco.loadImgs([ImgId])[0]

        annIds =  coco.getAnnIds(ImgId)
        anns = coco.loadAnns(annIds)
        for ann in anns:
            ann['category_id'] = ids_map[ann['category_id']]
            temp_annotation['annotations'].append(ann)
    print(anno_dir)
    print(anno_dir.replace('.json','_2cats.json'))
    with open(anno_dir.replace('.json','_2cats.json'), 'w') as outfile:
        json.dump(temp_annotation, outfile)            

def generate_dental_roi_with_max_bounds(anno_dir,imgs_dir):
    
    temp_annotation = json.load(open(anno_dir))
    coco = COCO(anno_dir)
    temp_annotation["images"] = []
    max_ann_id = 0
    #for ann in temp_annotation["annotations"]:
    #    if max_ann_id < ann["id"]:
    #        max_ann_id = ann["id"]
    print(max_ann_id)
    temp_annotation["annotations"] = []

    roi_cat_des = {"id": 1, "name": "ROI", "supercategory": ""}
    temp_annotation['categories'] = []
    temp_annotation['categories'].append(roi_cat_des)

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

        
        image = cv2.imread(os.path.join(imgs_dir,img['file_name']))
        
        if len(anns)>0:

            bound_box = [int(bound_x1),int(bound_y1),int(bound_x2),int(bound_y2)]
        
        else:

            bound_box = [0, 0, img['width'], img['height']]
        
        max_ann_id+=1
        
        roi_ann = {"id": max_ann_id, 
                    "image_id": ImgId, 
                    "category_id": roi_cat_des["id"], 
                    "segmentation": [],
                    "area": img['width']*img['height'],
                    "bbox": [], 
                    "iscrowd": 0,}

        #cropped_image = image[bound_box[1]:bound_box[3],bound_box[0]:bound_box[2]]
        
        # set the size of new cropped images in the annotation with integer
        #img['width'] = int(bound_box[2]-bound_box[0])
        #img['height'] = int(bound_box[3]-bound_box[1])

        roi_bbox = [bound_box[0],bound_box[1],int(bound_box[2]-bound_box[0]),int(bound_box[3]-bound_box[1])]
        
        roi_ann['bbox'] = roi_bbox

        #roi_ann["segmentation"] = [[roi_bbox[0],roi_bbox[1]]]

        temp_annotation["images"].append(img)

        #for ann in anns:
            # fix the bounding box position according to the cropped coordination
            #ann['bbox'][0] = round(ann['bbox'][0]-bound_box[0],2)
            #ann['bbox'][1] = round(ann['bbox'][1]-bound_box[1],2)

            # fix the polygon coordination
            #for idx,xy in enumerate(ann['segmentation'][0]):
            #    ann['segmentation'][0][idx] = round(xy - bound_box[idx%2],2)
            
        #    temp_annotation['annotations'].append(ann)
        
        temp_annotation["annotations"].append(roi_ann)

        #img_dir = os.path.join(croped_imgs_dir,img['file_name'])
        #os.makedirs(img_dir.replace(os.path.basename(img_dir),''),exist_ok=True)
        #cv2.imwrite(os.path.join(croped_imgs_dir,img['file_name']),cropped_image)            

    with open(anno_dir.replace('.json','_roi.json'), 'w') as outfile:
        json.dump(temp_annotation, outfile)  

if __name__=="__main__":
    # These are for crop1 dataset generating
    '''
    imgs_dir = "/home/qilei/.TEMP/TEETH3/images/"
    croped_imgs_dir = "/home/qilei/.TEMP/TEETH3/images_crop1/"

    anno_dir = "/home/qilei/.TEMP/TEETH3/annotations/train_1_3.json"
    dental_crop_with_max_bounds(anno_dir,imgs_dir,croped_imgs_dir)

    anno_dir = "/home/qilei/.TEMP/TEETH3/annotations/test_1_3.json"
    dental_crop_with_max_bounds(anno_dir,imgs_dir,croped_imgs_dir)   
    ''' 
    #ids_map = {1:1,2:2,3:2,4:2,5:2,6:2,7:2,8:2,9:2,10:2}
    #cat_names = {1:"normal",2:"abnormal"}
    
    #anno_dir = "/home/qilei/.TEMP/TEETH3/annotations/train_1_3.json"
    #redefine_id(ids_map,cat_names,anno_dir)

    #anno_dir = "/home/qilei/.TEMP/TEETH3/annotations/test_1_3.json"
    #redefine_id(ids_map,cat_names,anno_dir)


    imgs_dir = "/home/qilei/.TEMP/TEETH3/images/"
    #croped_imgs_dir = "/home/qilei/.TEMP/TEETH3/images_crop1/"

    anno_dir = "/home/qilei/.TEMP/TEETH3/annotations/train_1_3.json"
    generate_dental_roi_with_max_bounds(anno_dir,imgs_dir)

    anno_dir = "/home/qilei/.TEMP/TEETH3/annotations/test_1_3.json"
    generate_dental_roi_with_max_bounds(anno_dir,imgs_dir)  


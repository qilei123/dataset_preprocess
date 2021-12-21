from pycocotools.coco import COCO
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pylab
import cv2
import glob
import json
import os
from shapely.geometry import Point, Polygon, MultiPoint

def seg2points(seg):
    points = []
    
    for i in range(len(seg)):
        if i%2==1:
            points.append((seg[i-1],seg[i]))

    return points

def seg2minrect(seg):
    #seg = ann['segmentation'][0]
    #print(seg)
    polygon_points = seg2points(seg)
    polygon = Polygon(polygon_points)
    #print(polygon)
    min_rect = polygon.minimum_rotated_rectangle
    #print(min_rect)
    xs,ys = min_rect.exterior.coords.xy
    xys = []
    for i in range(4):
        xys.append(round(xs[i],2))
        xys.append(round(ys[i],2))
    return xys

def trans_merge_anns3(annotation_list,new_cats,save_dir,obb=False):
    
    image_names = list()
    img_id = 1
    anno_id = 1
    temp_annotation = json.load(open('annotations/split_annos/instances_default1.json'))

    temp_categories = []
    
    cat_id_count=0

    for new_cat in new_cats:
        cat_id_count+=1
        temp_cat = temp_annotation['categories'][cat_id_count]
        temp_cat['id'] = cat_id_count
        temp_cat['name'] = new_cat
        temp_categories.append(temp_cat)

    temp_annotation['categories'] = temp_categories

    temp_annotation["images"] = []
    temp_annotation["annotations"] = []
    count_ann_ids = dict()
    for annotation in annotation_list:
        coco=COCO(annotation['ann_dir'])

        for ImgId in coco.getImgIds():
            img = coco.loadImgs([ImgId])[0]
            #print(img)
            img['id'] = img_id
            temp_annotation["images"].append(img)
            if not(img['file_name'] in image_names):
                image_names.append(img['file_name'])
                annIds =  coco.getAnnIds(ImgId)
                anns = coco.loadAnns(annIds)
                img['file_name'] = os.path.join(annotation['ann_img_folder'],img['file_name'])
                for ann in anns:
                    if ann['category_id'] in annotation['ann_cat_ids_map']:
                        ann['category_id']=annotation['ann_cat_ids_map'][ann['category_id']]
                        ann['id'] = anno_id
                        ann['image_id'] = img_id
                        if obb:
                            #print(len(ann['segmentation'][0]))
                            if ann['category_id'] in count_ann_ids:
                                count_ann_ids[ann['category_id']]+=1
                            else:
                                count_ann_ids[ann['category_id']] = 1
                            if len(ann['segmentation'][0])<8:
                                continue

                            ann['segmentation']=[seg2minrect(ann['segmentation'][0])]
                        temp_annotation["annotations"].append(ann)
                        anno_id+=1
                    
            img_id+=1
    print(count_ann_ids)
    print(len(temp_annotation["images"]))
    print(len(temp_annotation["annotations"]))

    with open(save_dir, 'w') as outfile:
        json.dump(temp_annotation, outfile)

def regist_annotation(ann_dir,ann_cat_ids_map,ann_img_folder):
    temp_dict = dict()
    temp_dict['ann_dir'] = ann_dir
    temp_dict['ann_cat_ids_map'] = ann_cat_ids_map
    temp_dict['ann_img_folder'] = ann_img_folder
    return temp_dict

def merge():
    annotation_list = []
    
    ann1_dir = "trans_drone_andover_annotations/test_mix.json"
    ann1_cat_ids_map = {1:1,2:2,3:3}
    ann1_img_folder = 'andover'
    annotation_list.append(regist_annotation(ann1_dir,ann1_cat_ids_map,ann1_img_folder))

    ann2_dir = "trans_drone_woster_annotations/val_anns_p1.json"
    ann2_cat_ids_map = {1:1,2:2,3:3}
    ann2_img_folder = 'worcester'
    annotation_list.append(regist_annotation(ann2_dir,ann2_cat_ids_map,ann2_img_folder))

    new_cats = ['Small 1-piece vehicle',
                        'Large 1-piece vehicle',
                        'Extra-large 2-piece truck',]

    save_dir = "trans_drone_annotations/test_AW_obb.json"
    
    trans_merge_anns3(annotation_list,new_cats,save_dir,True)

def DroneVehicle_cat_id(cats,cat_name):
    cat_name = cat_name.replace("_"," ")
    if not (cat_name in cats):
        return -1
    cat_id=0
    for cat in cats:
        cat_id+=1
        if cat_name==cat:
            return cat_id

def DroneVehicle_box_seg(xml_info,roi):
    if len(xml_info)==8:
        seg = []
        count=0
        minx = 10000
        miny = 10000
        maxx = -1
        maxy = -1
        for cood in xml_info:
            if count%2==0:
                cood_value = int(cood.text)-roi[0]
                if minx>cood_value:
                    minx = cood_value
                if maxx<cood_value:
                    maxx = cood_value
            else:
                cood_value = int(cood.text)-roi[1]
                if miny>cood_value:
                    miny = cood_value
                if maxy<cood_value:
                    maxy = cood_value
            seg.append(cood_value)
        
            count+=1
        bbox = [minx,miny,maxx-minx,maxy-miny]
        #print(bbox)
        return bbox,seg
    elif len(xml_info)==4:
        minx = int(xml_info[0].text)-roi[0]
        miny = int(xml_info[1].text)-roi[1]
        maxx = int(xml_info[2].text)-roi[0]
        maxy = int(xml_info[3].text)-roi[1]        
        bbox = [minx,miny,maxx-minx,maxy-miny]
        seg = [minx,miny,maxx,miny,maxx,maxy,minx,maxy]
        return bbox,seg

def DroneVehicle2COCO():
    
    set_name = "val"
    xml_folder = "VisDrone-DroneVehicle/"+set_name+"/"+set_name+"label"
    xml_file_list = glob.glob(os.path.join(xml_folder,"*.xml"))
    img_folder = "VisDrone-DroneVehicle/"+set_name+"/"+set_name+"img"

    croped_folder = "VisDrone-DroneVehicle/images"

    coco_save_dir = "VisDrone-DroneVehicle/annotations/"+set_name+".json"

    roi = [100,100,740,612]

    new_cats = ['car','truck','bus','van','feright car']

    img_id = 1
    anno_id = 1

    temp_anno_dir = 'annotations/split_annos/instances_default1.json'
    temp_annotation = json.load(open(temp_anno_dir))

    temp_categories = []
    
    cat_id_count=0

    for new_cat in new_cats:
        cat_id_count+=1
        temp_cat =  {"id": -1, "name": "", "supercategory": ""}
        temp_cat['id'] = cat_id_count
        temp_cat['name'] = new_cat
        temp_categories.append(temp_cat)

    temp_annotation['categories'] = temp_categories

    temp_annotation["images"] = []
    temp_annotation["annotations"] = []
    all_names = dict()
    for xml_file in xml_file_list:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        img_info = {"id": img_id, "width": 640, "height": 512, 
                    "file_name": "", 
                    "license": 0, "flickr_url": "", "coco_url": "", 
                    "date_captured": 0}
        img_dir = os.path.join(img_folder,os.path.basename(xml_file).replace("xml","jpg"))
        image = cv2.imread(img_dir)

        for level1 in root:
            if level1.tag=='filename':
                img_name = set_name+"/"+os.path.basename(xml_file).replace("xml","jpg")
                '''
                if not ".jpg" in img_name:
                    img_name+='.jpg'
                img_name=img_name.replace("/","")
                '''
                img_info['file_name'] = img_name
                image = image[roi[1]:roi[3],roi[0]:roi[2]]
                cv2.imwrite(os.path.join(croped_folder,img_name),image)
            elif level1.tag=='object':
                if not level1[0].text in all_names:
                    all_names[level1[0].text]=1
                else:
                    all_names[level1[0].text]+=1
                if len(level1)==5 and (len(level1[4])==8 or len(level1[4])==4):
                    bbox,seg = DroneVehicle_box_seg(level1[4],roi)
                    ann = {"id": anno_id, 
                        "image_id": img_id, 
                        "category_id": DroneVehicle_cat_id(new_cats,level1[0].text), 
                        "segmentation": [seg], 
                        "area": bbox[2]*bbox[3], 
                        "bbox": bbox, 
                        "iscrowd": 0, 
                        "attributes": {"occluded": False}}
                    temp_annotation["annotations"].append(ann)
                    anno_id+=1
        temp_annotation["images"].append(img_info)
        img_id+=1
        if img_id%1000==0:
            print(img_id)
    print(all_names)
    with open(coco_save_dir, 'w') as outfile:
        json.dump(temp_annotation, outfile)
if __name__=="__main__":

    merge()
    #DroneVehicle2COCO()
    pass
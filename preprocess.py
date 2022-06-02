import json
import os
import glob
from posixpath import join
from pycocotools.coco import COCO

def unzip_all(dir):
    zip_filelist = glob.glob(os.path.join(dir,"*.zip"))
    for zip_file in zip_filelist:
        save_folder = os.path.basename(zip_file).replace(".zip","")
        #os.makedirs(save_folder)
        #command1 = "unzip "+zip_file#+" -d "+save_folder
        #os.system(command1)
        #command2 = "mv "+save_folder+"/images ./"
        #os.system(command2)
        print(save_folder)
        ann_folder = os.path.join("annotations",save_folder)
        os.makedirs(ann_folder)
        command3 = "cp "+save_folder+"/annotations/instances_default.json "+ann_folder
        os.system(command3)

def get_all_json_dirs(dir,key_string="2021"):
    folder_dir_list = glob.glob(os.path.join(dir,"*"+key_string+"*"))
    return [os.path.join(folder_dir,"instances_default.json") for folder_dir in folder_dir_list]

def create_cat_id_map(annotation,
                    new_cat_list=["Normal","Caries","Implant","Inlay","Pontic","Residual root",
                                "Retainer","Teeth filling","Veneer","Crown"]):
    cat_id_map = dict()
    
    assert len(new_cat_list)==len(annotation['categories']),"Number of category does not match!"

    for category in annotation['categories']:
        new_id = new_cat_list.index(category['name'])+1
        cat_id_map[category['id']] = new_id

    return cat_id_map

def update_cat_ids(annotation,cat_id_map):
    #update category id in categories
    for category in annotation['categories']:
        category['id'] = cat_id_map[category['id']]
    #update category id in annotations
    for ann in annotation['annotations']:
        ann['category_id'] = cat_id_map[ann['category_id']]

def update_cat_ids_and_save(instance_default_dir_list):
    for instance_default_dir in instance_default_dir_list:
        updated_instance_default_dir = instance_default_dir.replace(".json","_update.json")
        
        if os.path.exists(updated_instance_default_dir):
            continue

        annotation = json.load(open(instance_default_dir))
        update_cat_ids(annotation,create_cat_id_map(annotation))
        
        with open(updated_instance_default_dir, 'w') as outfile:
            json.dump(annotation, outfile)        

def merge2(annotation1,annotation2):
    int_image_id = 1
    ini_ann_id = 1

    images = []
    annotations = []

def merge_instances_defaults_t1(src_dir,save_dir):

    instance_default_dir_list = get_all_json_dirs("annotations/t2")
    
    update_cat_ids_and_save(instance_default_dir_list)

    temp_annotation = json.load(open('annotations/t2/instances_default.json'))
    update_cat_ids(temp_annotation,create_cat_id_map(temp_annotation))

    img_id = 1
    anno_id = 1    

    images = []
    annotations = []

    image_names = []

    for instance_default_dir in instance_default_dir_list:
        updated_instance_default_dir = instance_default_dir.replace(".json","_update.json")
        coco=COCO(updated_instance_default_dir)
        for ImgId in coco.getImgIds():
            img = coco.loadImgs([ImgId])[0]
            img['id'] = img_id
            images.append(img)
            image_name = os.path.basename(img['file_name'])
            if not(image_name in image_names):
                image_names.append(img['file_name'])
                annIds =  coco.getAnnIds(ImgId)
                anns = coco.loadAnns(annIds)
                for ann in anns:
                    ann['id'] = anno_id
                    ann['image_id'] = img_id
                    annotations.append(ann)
                    anno_id+=1
            img_id+=1

    temp_annotation["images"] = images
    temp_annotation["annotations"] = annotations

    with open(save_dir, 'w') as outfile:
        json.dump(temp_annotation, outfile)    

def update_cat_id_t1():
    json_files = glob.glob("annotations/t1/*.json")
    update_cat_ids_and_save(json_files)

def update_t1_from_t2():

    t1_train_dir = "annotations/t1/train_update.json"
    t1_train_json = json.load(open(t1_train_dir))
    t1_train_coco = COCO(t1_train_dir)

    t1_train_img_id_map = dict()
    for img in t1_train_json['images']:
        t1_train_img_id_map[os.path.basename(img['file_name'])] = img['id']

    t1_test_dir = "annotations/t1/test_update.json"
    t1_test_json = json.load(open(t1_test_dir))
    t1_test_coco = COCO(t1_test_dir)

    t1_test_img_id_map = dict()
    for img in t1_test_json['images']:
        t1_test_img_id_map[os.path.basename(img['file_name'])] = img['id']

    t2_dir = "annotations/t2/total2.json"
    t2_json = json.load(open(t2_dir))
    t2_coco = COCO(t2_dir)
    
    t2_img_id_map = dict()
    for img in t2_json['images']:
        file_name = os.path.basename(img['file_name'])
        t2_img_id_map[file_name] = img['id']  
        assert ((file_name in t1_train_img_id_map) or (file_name in t1_test_img_id_map)), "No such image to update:"+file_name

    train_temp_annotation = json.load(open('annotations/t2/instances_default.json'))
    update_cat_ids(train_temp_annotation,create_cat_id_map(train_temp_annotation))

    train_img_id = 1
    train_anno_id = 1    

    train_images = []
    train_annotations = []    

    for ImgId in t1_train_coco.getImgIds():
        img = t1_train_coco.loadImgs([ImgId])[0]
        img['id'] = train_img_id
        train_images.append(img)
        image_name = os.path.basename(img['file_name'])
        if image_name in t2_img_id_map:
            ImgId = t2_img_id_map[image_name]
            annIds =  t2_coco.getAnnIds(ImgId)
            anns = t2_coco.loadAnns(annIds)
            for ann in anns:
                ann['id'] = train_anno_id
                ann['image_id'] = train_img_id
                train_annotations.append(ann)
                train_anno_id+=1            
        else:
            annIds =  t1_train_coco.getAnnIds(ImgId)
            anns = t1_train_coco.loadAnns(annIds)
            for ann in anns:
                ann['id'] = train_anno_id
                ann['image_id'] = train_img_id
                train_annotations.append(ann)
                train_anno_id+=1
        train_img_id+=1

    train_temp_annotation["images"] = train_images
    train_temp_annotation["annotations"] = train_annotations

    train_save_dir = "annotations/t1_from_t2/train_updated.json"
    with open(train_save_dir, 'w') as outfile:
        json.dump(train_temp_annotation, outfile)  

    test_temp_annotation = json.load(open('annotations/t2/instances_default.json'))
    update_cat_ids(test_temp_annotation,create_cat_id_map(test_temp_annotation))

    test_img_id = 1
    test_anno_id = 1    

    test_images = []
    test_annotations = []    

    for ImgId in t1_test_coco.getImgIds():
        img = t1_test_coco.loadImgs([ImgId])[0]
        img['id'] = test_img_id
        test_images.append(img)
        image_name = os.path.basename(img['file_name'])
        if image_name in t2_img_id_map:
            ImgId = t2_img_id_map[image_name]
            annIds =  t2_coco.getAnnIds(ImgId)
            anns = t2_coco.loadAnns(annIds)
            for ann in anns:
                ann['id'] = test_anno_id
                ann['image_id'] = test_img_id
                test_annotations.append(ann)
                test_anno_id+=1            
        else:
            annIds =  t1_test_coco.getAnnIds(ImgId)
            anns = t1_test_coco.loadAnns(annIds)
            for ann in anns:
                ann['id'] = test_anno_id
                ann['image_id'] = test_img_id
                test_annotations.append(ann)
                test_anno_id+=1
        test_img_id+=1

    test_temp_annotation["images"] = test_images
    test_temp_annotation["annotations"] = test_annotations

    test_save_dir = "annotations/t1_from_t2/test_updated.json"
    with open(test_save_dir, 'w') as outfile:
        json.dump(test_temp_annotation, outfile)  

def get_instance_count(json_dir,cat_names=["Normal","Caries","Implant","Inlay","Pontic","Residual root",
                                "Retainer","Teeth filling","Veneer","Crown"]):
    coco = COCO(json_dir)
    print(json_dir)
    for cat_name in cat_names:
        catId = coco.getCatIds([cat_name])
        annids = coco.getAnnIds(catIds = catId)
        print(cat_name+":"+str(len(annids)))
if __name__=="__main__":
    #unzip_all("./")
    #merge_instances_defaults_t1("annotations/t2","annotations/t2/total2.json")
    #update_cat_id_t1()
    #update_t1_from_t2()
    get_instance_count("annotations/t1/train.json")
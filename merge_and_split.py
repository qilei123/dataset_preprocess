import json
from pycocotools.coco import COCO

def merge():
    temp_annotation = json.load(open('annotations/t1_from_t2/test_updated.json'))

    j1 = json.load(open('annotations/t1_from_t2/test_updated.json'))
    j2 = json.load(open('annotations/t1_from_t2/train_updated.json'))

    temp_annotation["images"] = []
    temp_annotation["annotations"] = []

    img_id = 1
    anno_id = 1

    coco1 = COCO('annotations/t1_from_t2/test_updated.json')

    coco2 = COCO('annotations/t1_from_t2/train_updated.json')

    for coco in [coco1,coco2]:
        for ImgId in coco.getImgIds():
            
            img = coco.loadImgs([ImgId])[0]
            img['id'] = img_id
            temp_annotation["images"].append(img)
            
            annIds =  coco.getAnnIds(ImgId)
            anns = coco.loadAnns(annIds)

            for ann in anns:
                ann['id'] = anno_id
                ann['image_id'] = img_id  
                temp_annotation["annotations"].append(ann)
                anno_id+=1

            img_id+=1

    with open('annotations/t1_from_t2/all.json', 'w') as outfile:
        json.dump(temp_annotation, outfile)

def split_half():
    
    temp_annotation1 = json.load(open('annotations/t1_from_t2/all.json'))
    temp_annotation1["images"] = []
    temp_annotation1["annotations"] = []

    temp_annotation2 = json.load(open('annotations/t1_from_t2/all.json'))
    temp_annotation2["images"] = []
    temp_annotation2["annotations"] = []
    
    coco = COCO('annotations/t1_from_t2/all.json')
    
    count=0
    
    for ImgId in coco.getImgIds():
        
        if count%2==1:
            temp_annotation = temp_annotation1
        else:
            temp_annotation = temp_annotation2

        img = coco.loadImgs([ImgId])[0]

        temp_annotation["images"].append(img)
        
        annIds =  coco.getAnnIds(ImgId)
        anns = coco.loadAnns(annIds)

        for ann in anns:
            temp_annotation["annotations"].append(ann)  
        count+=1
    with open('annotations/t1_from_t2/half1.json', 'w') as outfile:
        json.dump(temp_annotation1, outfile)
    with open('annotations/t1_from_t2/half2.json', 'w') as outfile:
        json.dump(temp_annotation2, outfile)

def generate_test():
    temp_annotation1 = json.load(open('annotations/t1_from_t2/half1.json'))
    temp_annotation1["images"] = []
    temp_annotation1["annotations"] = []    

    coco = COCO('annotations/t1_from_t2/half1.json')

    for ImgId in coco.getImgIds()[:20]:
        print(ImgId)        

if __name__=="__main__":
    #split_half()
    generate_test()
                  
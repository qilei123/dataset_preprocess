#from curses import has_colors
import os
import glob
from turtle import width
from webbrowser import get
import zipfile
import xml.etree.ElementTree as ET
import cv2
import sys
import random
import numpy as np
from pycocotools.coco import COCO

sys.path.append("D:\\DEVELOPMENT\\train_img_classifier")



def change_video_names():
    records = open('temp_datas/changweijing_issues1.txt',encoding="utf8")
    #map_records = open('temp_datas/changweijing_issues_filenames_map.txt','w',encoding="utf8")
    line = records.readline()
    count=0
    new_base_name = '20220301_1024_010'

    video_dir = '/data3/qilei_chen/DATA/changjing_issues/'

    while line:
        
        line = line.replace('\n','')
        line_eles = line.split('	')

        if len(line_eles[0])>0:
            new_line = line_eles[0]+" "+new_base_name+str(count)+"_"+line_eles[1]
            new_line_eles = new_line.split(' ')
            
            if os.path.exists(os.path.join(video_dir,new_line_eles[0]+".avi")):
                org_video_name = os.path.join(video_dir,new_line_eles[0]+".avi")
            else:
                org_video_name = os.path.join(video_dir,new_line_eles[0]+".mp3")
  
            os.rename(org_video_name,org_video_name.replace(new_line_eles[0],new_line_eles[1]))
            print(org_video_name)
            print(org_video_name.replace(new_line_eles[0],new_line_eles[1]))

            #map_records.write(new_line)
            count+=1

        line = records.readline()
    

def change_video_names1():
    root_dir = '/data3/qilei_chen/DATA/changjing_issues/'
    video_dirs = glob.glob(os.path.join(root_dir,'*mp4'))+glob.glob(os.path.join(root_dir,'*avi'))

    map_name_records = open(os.path.join(root_dir,'video_names_map_bk.txt'))

    #new_base_name = '20220301_1024_010'

    for count,video_dir in enumerate(video_dirs):
        video_name = os.path.basename(video_dir)
        #new_video_dir = os.path.join(root_dir,new_base_name+str(count)+'.'+video_dir[-3:])

        new_video_dir = video_dir.replace(video_dir[-4:],"."+video_dir[-3:])
        #map_name_records.write(video_name+' '+new_video_name+'\n')
        os.rename(video_dir,new_video_dir)
        print(video_dir)
        print(new_video_dir)

def change_video_names2():
    root_dir = '/data3/qilei_chen/DATA/changjing_issues/'
    video_dirs = glob.glob(os.path.join(root_dir,'*mp4'))+glob.glob(os.path.join(root_dir,'*avi'))

    map_name_records = open(os.path.join(root_dir,'video_names_map_bk.txt'))

    #new_base_name = '20220301_1024_010'
    line = map_name_records.readline()

    while line:
        eles = line[:-1].split(' ')
        eles[1] = eles[1].replace('_mp4','.mp4')
        eles[1] = eles[1].replace('_avi','.avi')
        file_dir = os.path.join(root_dir,eles[1])
        o_file_dir = os.path.join(root_dir,eles[0])
        if os.path.exists(file_dir):
            os.rename(file_dir,o_file_dir)
        line = map_name_records.readline()

def unzip_files(file_dir):
    file_list = glob.glob(os.path.join(file_dir,"*.zip"))
    for file in file_list:
        with zipfile.ZipFile(file,"r") as zip_ref:
            zip_ref.extractall(file.replace(".zip",""))        

def parse_xml_ann(folder_dir,record = None) :
    folder_list = glob.glob(os.path.join(folder_dir,"*[!.zip]"))
    layer1_labels = ["NBI+放大","白光","NBI+非放大"]
    layer2_labels = ["萎缩性胃炎","肠化","低级别","高级别","癌变","低级别+高级别"]

    count_matrix = [[0 for x in range(len(layer2_labels))] for y in range(len(layer1_labels))]


    for folder in folder_list:
        #print(os.path.basename(folder))
        xml_dir  = os.path.join(folder,"annotations.xml")
        
        xml_tree = ET.parse(xml_dir)

        xml_root = xml_tree.getroot()

        for child in xml_root.iter("image"):

            tags = child.findall("tag")
            #print(child.attrib["name"])
            layer1_index = -1
            layer2_index = -1

            for tag in tags:
                if tag.attrib["label"] in layer1_labels and layer1_index < layer1_labels.index(tag.attrib["label"]):
                    
                    layer1_index = layer1_labels.index(tag.attrib["label"])

                if tag.attrib["label"] in layer2_labels and  layer2_index < layer2_labels.index(tag.attrib["label"]):
                    layer2_index = layer2_labels.index(tag.attrib["label"])

            if layer1_index>=0 and layer2_index>=0:
                count_matrix[layer1_index][layer2_index]+=1
                record.write(os.path.join(folder,"images",child.attrib["name"])+" "+os.path.basename(folder)+" "+layer1_labels[layer1_index]+" "+layer2_labels[layer2_index]+"\n")
    print(count_matrix)

def tally_fangdaweijing():
    
    folder_dir_list = ["E:\\DATASET\\放大胃镜\\放大胃镜图片筛选\\2015_放大胃镜标注\\",
                            "E:\\DATASET\\放大胃镜\\放大胃镜图片筛选\\2017_放大胃镜标注\\",
                            "E:\\DATASET\\放大胃镜\\放大胃镜图片筛选\\2020\\",
                            "E:\\DATASET\\放大胃镜\\放大胃镜图片筛选\\2021\\"]
    
    record = open("E:/DATASET/放大胃镜/放大胃镜图片筛选/2015_2017_2020_2021_list.log", "w", encoding="utf-8")

    for folder_dir in folder_dir_list:
        for data_dir in glob.glob(folder_dir+"*"):
            print(data_dir)
            #unzip_files(data_dir)
            parse_xml_ann(data_dir, record)

def get_adenoma(data_dir):
    record_file = open(os.path.join(data_dir, "test.txt"), "r")

    record_line = record_file.readline()

    label_dic =  {0:"adenoma",1:"noneadenoma"}

    while record_line:
        
        src_file_dir = record_line[:-3]
        label = int(record_line[-2])

        target_dir = os.path.join(data_dir,"val",label_dic[label])

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        command = "cp "+os.path.join(data_dir,"test",src_file_dir)+" "+target_dir
        record_line = record_file.readline()
def generate_clip_frames(video_dir,period=[0,200]) :
    from img_crop import crop_img
    cap = cv2.VideoCapture(video_dir)

    frameRate = int(cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    save_name = video_dir[:-4] + "_crop.avi"
    

    save_dir = video_dir.replace(".avi","")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    success,frame = cap.read()

    frame_index=0
    frame,roi = crop_img(frame)
    outputCap = cv2.VideoWriter(save_name, codec, frameRate, (int(roi[2]-roi[0]),int(roi[3]-roi[1])) )

    while success:

        #if frame_index>period[0] and frame_index<period[1] :
        #    cv2.imwrite(os.path.join(save_dir,str(frame_index)+".jpg"),frame)

        frame_index+=1

        #if frame_index>period[1]:
        #    break
        outputCap.write(frame)
        success,frame = cap.read()
        if success:
            frame = crop_img(frame,roi)
        

def random_rename(folder_dir1,folder_dir2):

    file_list1 = glob.glob(os.path.join(folder_dir1,"*"))
    file_list2 = glob.glob(os.path.join(folder_dir2,"*"))
    
    file_list = file_list1 + file_list2
    random.shuffle(file_list)

    file_map_record1 = open(folder_dir1+".txt", "w",encoding="utf-8")
    file_map_record2 = open(folder_dir2+".txt", "w",encoding="utf-8")

    for index,file in enumerate(file_list):
        if folder_dir1 in file:
            folder_dir = folder_dir1
            file_map_record = file_map_record1
        else:
            folder_dir = folder_dir2
            file_map_record = file_map_record2
        os.rename(file,os.path.join(folder_dir,str(index).zfill(5)+".jpg"))
        file_map_record.write(os.path.basename(file)+"  "+str(index).zfill(5)+".jpg\n")


def generate_fangdaweijing(folder_dir,file_names, key_word = 'NBI+放大'):

    labels = ["萎缩性胃炎","肠化","低级别","高级别","癌变"]
    label_ids = [0,0,1,2,3]

    records = dict()

    for file_name in file_names:

        record_file = open(os.path.join(folder_dir,file_name),encoding="utf-8")

        line = record_file.readline()

        while line:
            eles = line.split(' ')
            if len(eles)==2:
                if key_word in eles[0]:
                    key_id = eles[1][:-2]
                    if not key_id in records:
                        records[key_id] = []
                    records[key_id].append(eles[0])
            elif len(eles)==4:
                if key_word == eles[2]:
                    key_id = eles[1][:-2]
                    if not key_id in records:
                        records[key_id] = []
                    records[key_id].append([eles[0],eles[3]])                
            line = record_file.readline()
    count=0
    print(len(records))
    exit()
    train_record_file = open(os.path.join(folder_dir,'v3_baiguang1.txt'),'w',encoding="utf-8")
    test_record_file = open(os.path.join(folder_dir,'v3_baiguang2.txt'),'w',encoding="utf-8")
    for key_id in records:
        if count%4==0:
            record_file = test_record_file
        else:
            record_file = train_record_file
        for image_file in records[key_id]:
            if isinstance(image_file,str):
                for label_index, label in enumerate(labels) :
                    
                        if label in image_file:
                            
                            record_file.write(image_file+' '+str(label_ids[label_index])+"\n")    
                            break             
            elif isinstance(image_file,list):  
                #print(image_file) 
                for label_index, label in enumerate(labels) : 
                    
                    if label == image_file[1][:-1]:
                        
                        record_file.write(image_file[0]+' '+str(label_ids[label_index])+"\n")
                        break
            
        count+=1

def CropImg(image,roi=None):
    if roi is None:
        height, width, d = image.shape

        pixel_thr = 10
        
        w_start=0
        while True:
            if np.sum(image[int(height/2),int(w_start),:])/d>pixel_thr:
                break
            w_start+=1

        w_end=int(width-1)
        while True:
            if np.sum(image[int(height/2),int(w_end),:])/d>pixel_thr:
                break
            w_end-=1

        h_start=0
        while True:
            if np.sum(image[int(h_start),int(width/2),:])/d>pixel_thr:
                break
            h_start+=1

        h_end=int(height-1)
        while True:
            if np.sum(image[int(h_end),int(width/2),:])/d>pixel_thr:
                break
            h_end-=1

        roi = [w_start,h_start,w_end,h_end]

        #print(image[int(height-1),int(width-1),:])

    return image[roi[1]:roi[3],roi[0]:roi[2],:],roi


def crop_abn_FD(src_dir,dst_dir):
    file_list = glob.glob(os.path.join(src_dir,"*.jpg"))
    roi = None
    for file_dir in file_list:
        image = cv2.imdecode(np.fromfile(file_dir, dtype=np.uint8), -1)#cv2.imread(file_dir)
        #image,roi = crop_img(image)
        if roi is None:
            crop_image,roi = CropImg(image)
        else:
            crop_image,roi = CropImg(image,roi)
        #print(roi)
        #print(crop_image)
        #cv2.imwrite(os.path.join(dst_dir,os.path.basename(file_dir)),crop_image)
        cv2.imencode('.jpg', crop_image)[1].tofile(os.path.join(dst_dir,os.path.basename(file_dir)))


def get_baiguang_images():
    record_names = ['v3_baiguang1','v3_baiguang2']
    root_dir = 'E:/DATASET/放大胃镜/放大胃镜图片筛选/'

    for record_name in record_names:
        print(record_name)
        records = open(root_dir+record_name+'.txt',encoding="utf-8")

        record = records.readline()

        while record:
            record = record.split(' ')
            record[0] = record[0].replace("\\",'/')
            if os.path.exists(os.path.join(root_dir,record[0])):
                try:
                    image = cv2.imdecode(np.fromfile(os.path.join(root_dir,record[0]), dtype=np.uint8), -1)

                    save_dir = os.path.join(root_dir, record_name, str(record[1][:-1]))

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    #cv2.imwrite(os.path.join(save_dir, os.path.basename(record[0])), image)
                    cv2.imencode('.jpg', image)[1].tofile(os.path.join(save_dir, os.path.basename(record[0])))
                except:
                    print(record[0])

            else:
                print(os.path.join(root_dir,record[0]))

            record = records.readline()
    

def cropWithMask():
    coco = COCO("E:/DATASET/Dental/annotations/test_1_3_crop.json")

    ann_id = 19953

    file_name = coco.imgs[coco.anns[ann_id]["image_id"]]["file_name"]

    file_path = os.path.join+("E:/DATASET/Dental","images_crop1",file_name)

    file_path = file_path.replace("\\","/")

    print(file_path)

    image = cv2.imread(file_path)

    seg = coco.anns[ann_id]["segmentation"]

    pts = np.array(seg,dtype=int)
    pts = pts.reshape((-1,2))
    
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = image[y:y + h, x:x + w].copy()  

    cv2.imwrite("crop.jpg",croped)  
    
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    
    img = cv2.bitwise_and(croped, croped, mask=mask)
    
    cv2.imwrite("test.jpg",img)

def generateFoldConfusionMatrix():
    
    dataset_dir ='/home/qilei/.TEMP/FDWJ/v3_3'
    model_dir = 'work_dir/swin_base_patch4_window7_224-224/'
    record_name = 'swin_base_patch4_window7_224.csv'
    print(os.path.join(dataset_dir,model_dir,record_name))
    record_file = open(os.path.join(dataset_dir,model_dir,record_name))

    line = record_file.readline()    

    while line:
        print(line)
        eles = line.split(',')

        dst_dir = os.path.join(dataset_dir,model_dir,'confusion_matrix',eles[1],eles[2])

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        src_dir = os.path.join(dataset_dir,'test',eles[1],eles[0])

        command_str = 'cp '+ src_dir+" "+dst_dir

        os.system(command_str)

        line = record_file.readline()

import json
from pycocotools.coco import COCO
import os
import skimage.io as io
import matplotlib.pyplot as plt
def show_gt_wj():
    dataset_dir = "E:/DATASET/放大胃镜/放大胃镜图片筛选/v3_白光/"

    ann_dir = dataset_dir+'train.json'

    img_folder = dataset_dir

    coco = COCO(ann_dir)

    ImgId = 37

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

if __name__=="__main__":
    #change_video_names2()
    #tally_fangdaweijing()
    #get_adenoma("/data3/qilei_chen/DATA/polyp_xinzi/D1_D2")

    #generate_clip_frames("E:/DATASET/Camera_motion_estimation/20211027_1626_1631_c_2.42-5.23.avi")
    #random_rename("E:/DATASET/腺瘤&非腺瘤/val/none_adenoma","E:/DATASET/腺瘤&非腺瘤/val/adenoma")

    #generate_fangdaweijing('E:\DATASET\放大胃镜\放大胃镜图片筛选',['2016_match_lists.log','2018_match_lists.log','2019_match_lists.log','2015_2017_2020_2021_list.log'])
    
    #crop_abn_FD("E:/DATASET/放大胃镜/放大胃镜图片筛选/abnormal_roi_images/org2/4","E:/DATASET/放大胃镜/放大胃镜图片筛选/abnormal_roi_images/org2/4_crop")
    #get_baiguang_images()
    #cropWithMask()

    
    generateFoldConfusionMatrix()
    pass

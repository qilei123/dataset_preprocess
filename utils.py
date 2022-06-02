import os
import glob
import zipfile
import xml.etree.ElementTree as ET

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

def parse_xml_ann(folder_dir) :
    folder_list = glob.glob(os.path.join(folder_dir,"*[!.zip]"))
    layer1_labels = ["NBI+放大","白光","NBI+非放大"]
    layer2_labels = ["萎缩性胃炎","肠化","低级别","高级别","癌","低级别+高级别"]

    count_matrix = [[0 for x in range(len(layer2_labels))] for y in range(len(layer1_labels))]


    for folder in folder_list:
        xml_dir  = os.path.join(folder,"annotations.xml")

        xml_tree = ET.parse(xml_dir)

        xml_root = xml_tree.getroot()

        for child in xml_root.iter("image"):

            tags = child.findall("tag")

            layer1_index = -1
            layer2_index = -1

            for tag in tags:
                if tag.attrib["label"] in layer1_labels and layer1_index < layer1_labels.index(tag.attrib["label"]):
                    
                    layer1_index = layer1_labels.index(tag.attrib["label"])

                if tag.attrib["label"] in layer2_labels and  layer2_index < layer2_labels.index(tag.attrib["label"]):
                    layer2_index = layer2_labels.index(tag.attrib["label"])

            if layer1_index>=0 and layer2_index>=0:
                count_matrix[layer1_index][layer2_index]+=1

    print(count_matrix)

def tally_fangdaweijing():
    data_dir = "E:\\DATASET\\放大胃镜\\放大胃镜图片筛选\\2020\\20220512"
    #unzip_files(data_dir)
    parse_xml_ann(data_dir)

if __name__=="__main__":
    #change_video_names2()
    tally_fangdaweijing()

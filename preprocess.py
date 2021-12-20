import json
import os
import glob
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


if __name__=="__main__":
    unzip_all("./")
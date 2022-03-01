import os
import glob
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

if __name__=="__main__":
    change_video_names2()
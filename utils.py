import os

def change_video_names():
    records = open('temp_datas/changweijing_issues1.txt',encoding="utf8")
    map_records = open('temp_datas/changweijing_issues_filenames_map.txt','w',encoding="utf8")
    line = records.readline()
    count=0
    new_base_name = '20220301_1024_010'
    while line:
        
        line = line.replace('\n','')
        line_eles = line.split('	')

        if len(line_eles[0])>0:
            new_line = line_eles[0]+" "+new_base_name+str(count)+"_"+line_eles[1]+"\n"
            map_records.write(new_line)
            count+=1

        line = records.readline()
    

if __name__=="__main__":
    change_video_names()
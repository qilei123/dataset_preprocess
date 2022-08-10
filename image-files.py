# -*- coding: UTF-8 -*-
import os
import re
import time
import shutil
from tqdm import tqdm


def info_extraction(doc_path):
    info_lists = []
    f = open(doc_path, 'r')
    for line in f.readlines():
        line = line.replace('\n', '')
        name = line.split('.')[0]
        info_lists.append(name)
    f.close()
    return info_lists


def file_extract_save(file_path, logs_path):
    output_path = logs_path + file_path.split('/')[-1] + '_images_lists.log'
    f = open(output_path, "w", encoding='utf-8')
    file_dirs = [os.path.join(file_path, x) for x in os.listdir(file_path)]
    file_lists = []
    for file_dir in file_dirs:
        for (dir_path, dir_names, file_names) in os.walk(file_dir):
            # file_lists += [os.path.join(dir_path, file).replace('\\', '/') for file in file_names]
            file_lists = [os.path.join(dir_path, file).replace('\\', '/') for file in file_names if file.endswith(('.jpg', '.jpeg', '.png'))]
            for file in file_lists:
                f.write(file[3:] + '\n')
    f.close()
    # return file_lists


def file_compare_plus(file_path, logs_path, store_path):
    output_path = store_path + file_path.split('-')[-2] + '_match_lists.log'
    f = open(output_path, "w", encoding='utf-8')
    r = open(logs_path, "r", encoding='utf-8')
    image_list = r.readlines()

    file_dirs = [os.path.join(file_path, x) for x in os.listdir(file_path)]
    file_lists = []
    for file_dir in file_dirs:
        for (dir_path, dir_names, file_names) in os.walk(file_dir):
            # file_lists += [os.path.join(dir_path, file).replace('\\', '/') for file in file_names]
            file_lists = [os.path.join(dir_path, file).replace('\\', '/') for file in file_names if file.endswith(('.jpg', '.jpeg', '.png'))]
            for file in tqdm(file_lists):
                for image in image_list:
                    image_name = image.split('/')[-1].replace('\n', '')
                    if image_name == file.split('/')[-1]:
                        f.write(image.replace('\n', '') + ' ' + str(file.split('/')[-3]) + '\n')
    r.close()
    f.close()


# 获取指定文件中文件名
def get_filename(file_type):
    name = []
    final_name_list = []
    source_dir = os.getcwd()  # 读取当前路径
    for root, dirs, files in os.walk(source_dir):
        for i in files:
            if file_type in i:
                name.append(i.replace(file_type, ''))
    final_name_list = [item + file_type for item in name]
    return final_name_list  # 返回由文件名组成的列表


# 筛选文件，利用正则表达式
def select_file(str_cond, file_name_list):
    select_name_list = []
    part1 = re.compile(str_cond)  # 正则表达式筛选条件
    for file_name in file_name_list:
        if len(part1.findall(file_name)):  # 判断其中一个文件名是否满足正则表达式的筛选条件
            select_name_list.append(file_name)  # 满足，则加入列表
    return select_name_list  # 返回由满足条件的文件名组成的列表


# 复制指定文件到另一个文件夹里，并删除原文件夹中的文件
def cope_file(select_file_name_list, old_path, new_path):
    for file_name in select_file_name_list:
        # 路径拼接要用os.path.join，复制指定文件到另一个文件夹里
        shutil.copyfile(os.path.join(old_path, file_name), os.path.join(new_path, file_name))
        os.remove(os.path.join(old_path, file_name))  # 删除原文件夹中的指定文件文件
    return select_file_name_list


# 主函数
def main_function(file_type, str_cond, old_path, new_path):
    final_name_list = get_filename(file_type)
    select_file_name_list = select_file(str_cond, final_name_list)
    cope_file(select_file_name_list, old_path, new_path)
    return select_file_name_list


if __name__ == '__main__':
    start_time = time.time()

    file_type = '.csv'  # 指定文件类型
    str_cond = '-Dfn_info-'  # 正则条件
    old_path = 'C:/Users/account/Desktop/Predict/'  # 原文件夹路径
    new_path = 'C:/Users/account/Desktop/Predict/'  # 新文件夹路径
    # main_function(file_type, str_cond, old_path, new_path)  # 主函数

    img_path = "E:/DATASET/放大胃镜/放大胃镜图片筛选/2016"
    out_path = "E:/"
    # file_extract_save(img_path, out_path)

    # image_log = '2018_images_lists.log'
    # save_path = 'D:/Datasets/trial&error/information/2018/'
    # docs_path = save_path + image_log
    # image_path = 'D:/Datasets/trial&error/information/2018/magnifying_endoscopy-2018-JPG/'
    # file_extract_save(image_path, docs_path)
    
    image_log = '2016_images_lists.log'
    save_path = 'F:/'
    docs_path = save_path + image_log
    image_path = 'F:/放大胃镜-2016-JPG/'
    file_compare_plus(image_path, docs_path, save_path)
    
    end_time = time.time()
    print('time_consuming: {:.2f} s'.format(end_time - start_time))

import os
import csv
from os import rename


# set path
split_info = 'Val'
path_root = '../src/data/h36m_eccv18_challenge/pose_2d/'
path_dir =  path_root + split_info + '/POSE'

# get file list from path
file_list = os.listdir(path_dir)
file_list.sort()

for item in file_list:
    # Write item to outcsv
    full_name = os.path.join(path_dir, item)
    full_name_new = os.path.join(path_dir, split_info + "_" + item)
    print (full_name)
    rename(full_name, full_name_new)

print("haha")


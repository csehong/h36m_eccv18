import os
import csv
from os import rename



# set path
split_info = 'Test'
path_root = '../src/data/h36m_eccv18/'
path_dir =  path_root + split_info + '/IMG'

# get file list from path
file_list = os.listdir(path_dir)
file_list.sort()

for item in file_list:
    # Write item to outcsv
    full_name = os.path.join(path_dir, item)
    full_name_new = os.path.join(path_dir, split_info + "_" + item)
    # print (full_name)
    rename(full_name, full_name_new)
    # print (full_name_new)

print("haha")

#
# # set path
# split_info = "Val"
# path_root = '../src/data/h36m_muzi'
#
#
# file_list = []
# split_path = os.path.join(path_root, "split", split_info + '_list.csv')
#
# with open(split_path, 'r') as f:
#     csvReader = csv.reader(f)
#     for row in csvReader:
#         # print (row)
#         file_list.append(row[0].split('.jp')[0])
#
# split_path = os.path.join(path_root, "split", split_info + '_list_new.csv')
#
# with open(split_path, 'a') as outcsv:
#     #configure writer to write standard csv file
#     writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
#     for file in file_list:
#         # print ([file])
#         writer.writerow([file])
#     # print (full_name_new)
import os
import csv


# set path
split_info = 'Test'
path_root = '../src/data/h36m_eccv18/' +split_info + '/IMG/'

# get file list from path
file_list = os.listdir(path_root)
file_list.sort()

# write file list to .csv
file_path = '../src/data/h36m_eccv18_challenge/split/' + split_info + "_list_cpm.csv"
with open(file_path, 'a') as outcsv:
    #configure writer to write standard csv file
    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    for item in file_list:
        #Write item to outcsv
        if item.split('_')[0] != split_info:
            continue
        item_short = item.split('.')[0]
        # print (item_short)
        writer.writerow([item_short])



print("haha")


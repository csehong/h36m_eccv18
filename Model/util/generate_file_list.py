import os
import csv


# set path
split_info = 'Train'
path_root = '../src/data/h36m_eccv18_challenge/img/'

# get file list from path
file_list = os.listdir(path_root)
file_list.sort()

# write file list to .csv
file_path = '../src/data/h36m_eccv18_challenge/split/' + split_info + "_list.csv"
with open(file_path, 'a') as outcsv:
    #configure writer to write standard csv file
    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    for item in file_list:
        #Write item to outcsv
        if item.split('_')[0] != split_info:
            continue
        print (item)
        writer.writerow([item])



print("haha")


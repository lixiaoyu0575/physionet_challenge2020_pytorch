import csv
import os
import wfdb
path = '/DATASET/challenge2020/All_data/'
def readname(filePath):
    name = os.listdir(filePath)
    name.sort()
    return name
file_colletion = readname(path)
dat_collection = []
for i in range(0,len(file_colletion)):
    if file_colletion[i].find('.mat')>=0:
        dat_collection.append(file_colletion[i].strip('.mat'))

f = open('Preprocess_Data/REFERENCE.csv','w',encoding='utf-8',newline='')
csv_writer = csv.writer(f)
for j in range(0, len(dat_collection)):
    print(dat_collection[j])
    record = wfdb.rdrecord(path + dat_collection[j])
    csv_writer.writerow([dat_collection[j], str(record.__dict__['comments'][2][4:])])
f.close()

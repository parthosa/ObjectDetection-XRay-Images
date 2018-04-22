import os
from scipy.io import loadmat
from math import ceil 

directory = '../Dataset-2/'
class_name = 'gun'
arr = []
for folder in os.listdir(directory):
    if not folder.startswith("."):
        if folder != 'B0013' and folder != 'B0031':
            bboxes = loadmat(directory+folder+'/BoundingBox.mat');
            files = os.listdir(directory+folder)
            for bb in bboxes['bb']:
                arr.append(['/data/'+folder+'/'+files[int(bb[0])-1],ceil(bb[1]),ceil(bb[2]),ceil(bb[3]),ceil(bb[4]),class_name])

import pickle

with open('datafile', 'wb') as fp:
    pickle.dump(arr, fp)


with open('datafile','rb') as f:

    # print('Parsing annotation files')
    itemlist = pickle.load(f)
    for line in itemlist:
        print line
# 	if folder
    # if i > 9 and i < 41 and i != 13 :
    # 	for files in os.listdir(directory+folder):
    # 		print files
    # if filename.endswith(".asm") or filename.endswith(".py"): 
    #     # print(os.path.join(directory, filename))
    #     continue
    # else:
    #     continue

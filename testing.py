import matplotlib.image
import os
import numpy as np
from glob import glob
def get_File(file_dir):
    # The images in each subfolder
    images = []
    # The subfolders
    subfolders = []
    # Using "os.walk" function to grab all the files in each folder
    dirPath=glob(file_dir+"\*")
    # for dirNames in dirPath:
    image_list=[]
    for dirPath, dirNames, fileNames in os.walk(file_dir):
        for labels in dirNames:
            file_list=glob(os.path.join(dirPath,labels)+"\*")
            image_list=image_list+file_list
            subfolders=subfolders+[labels]*len(file_list)
        # for name in fileNames:
        #     images.append(os.path.join(dirPath, name))
        # file_list=glob(dirNames+"\*")
        # images=images+file_list
        # subfolders=subfolders+[dirNames]*len(file_list)
        # print()
        # for name in dirNames:
        #     subfolders.append(os.path.join(dirPath, name))

    # To record the labels of the image dataset
    # labels = []
    # count = 0
    # for a_folder in subfolders:
    #     n_img = len(os.listdir(a_folder))
    #     labels = np.append(labels, n_img * [count])
    #     count+=1
    #
    # subfolders = np.array([images, labels])
    # subfolders = subfolders.transpose()
    #
    # image_list = list(subfolders[:, 0])
    # label_list = list(subfolders[:, 1])
    # label_list = [int(float(i)) for i in label_list]
    return image_list, subfolders
def one_hot(labels):
    label2one_hot={}
    one_hot2label=[]
    one_hotLabel=[]
    for l in labels:
        if l in label2one_hot:
            one_hotLabel.append(label2one_hot[l])
        else :
            label2one_hot[l]=len(one_hot2label)
            one_hotLabel.append(len(one_hot2label))
            one_hot2label.append(l)
    return label2one_hot,one_hot2label,one_hotLabel
image_dir,subfolders=get_File("train")
f2l,l2f,labels=one_hot(subfolders)
print("")
from os import *
import sys
import numpy as np
import cv2
import pandas as pd

""" 0 = palm, 1 = L, 2 = first, 3 = fist_moved, 4 = thumb, 5 = index 6 = OK"""
""" 7 = palm_moved, 8 = c, 9 = down"""


def color_image_gray(folder,filename):
    """ Read an image from a folder, color it gray and inver it """
    image = cv2.imread(path.join(folder,filename))
    image2 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return ~image2

def convert_image_into_binary_image(img):
    """ Conver a given image into binary so it be read by the computer """
    ret,bw_img = cv2.threshold(img,0,255, (cv2.THRESH_BINARY + cv2.THRESH_OTSU))
    return bw_img

def find_countours(image):
    """ Find the counturs in an image """
    return cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


def read_dataset(folder):
    """ Read a whole folder with images, resize them and save them as array """
    train_data = []
    for file in listdir(folder):
        img = color_image_gray(folder,file)
        img = cv2.resize(img,(int(160),int(60)))

        img = convert_image_into_binary_image(img)
        img = np.array(img).reshape(9600)
        train_data.append(img)
    train_data = np.array(train_data)
    return train_data

def create_dataset(folders):

    """ Create an array with all data points, append label and save them as array """

    p = "leapGestRecog/"
    outher_folder = 10
    inner_folder  = 10
    dataset = np.array([])
    for k in range(outher_folder):
        folder = folders[k]
        for i in range(inner_folder):
            data = read_dataset(p+"0"+str(i)+"/"+folder)
            data = np.insert(data,data.shape[1],[str(i)],axis=1)
            if k == 0 and i == 0:
                dataset = data
            else:
                dataset = np.concatenate((dataset,data))
            print("Step: [{},{}]".format(outher_folder*inner_folder, (k * inner_folder) + i + 1))

    print("final dataset shape is: " + str(dataset.shape))
    return dataset

def dataloaderMain():
    """ Create the dataset saved as a csv file name database.csv """

    dirs = listdir("leapGestRecog/00")
    folders = sorted(dirs)

    data = create_dataset(folders)
    data = pd.DataFrame(data,index=None)
    data.to_csv('data/database.csv',index=False)

dataloaderMain()

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:57:15 2019

@author: luket
"""
#https://www.tensorflow.org/tutorials/keras/basic_classification
from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
import os
import PIL
from matplotlib import image
import time as time
import pickle
from keras import metrics
import random
import pickle
import os
#end import section
#generate directory names for classes
dirs = os.listdir("root_data/images/Images")[:]
#for each dog breed (grouped by directory)
for oneDir in dirs:
    #adjust the actual folder name to index 
    oneDir = "root_data/images/Images/"+str(oneDir)
    #get all the image names
    imageNames = (os.listdir(oneDir))
    #loop through each image
    for imgName in imageNames:
        #grab the image
        img =  PIL.Image.open(oneDir + "/" + imgName)
        print(oneDir + "/" + imgName)
        #get the bounding box annotation
        f = open('root_data/Annotation/' + oneDir[24:] + "/"+imgName[:-4], "r")
        text = f.readlines()
        xmin = 0
        xmax = 250
        ymin = 0
        ymax = 250
        for line in text:
            if(line[4:8]=="xmin"):
                xmin = int(line[9:len(line)-8])
            elif(line[4:8]=="xmax"):
                xmax = int(line[9:len(line)-8])
            elif(line[4:8]=="ymin"):
                ymin = int(line[9:len(line)-8])
            elif(line[4:8]=="ymax"):
                ymax = int(line[9:len(line)-8])
        f.close()
        #crop the image
        cropped_img = img.crop((xmin,ymin,xmax,ymax))
        #save the image
        #cropped_img.save(oneDir + "/" + imgName)
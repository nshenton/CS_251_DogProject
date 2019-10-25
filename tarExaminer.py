# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:57:15 2019

@author: luket
"""
#https://www.tensorflow.org/tutorials/keras/basic_classification
from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
import os
import PIL
from matplotlib import image
import time as time
#root directories
rootDir = "root_data/images/n02098286-West_Highland_white_terrier/"
rootDir2 = "root_data/images/n02105641-Old_English_sheepdog/"
#store in array
dirs = [rootDir,rootDir2]
i = 0
#initialize matrix of all photos
dogPictureMatrix = np.zeros((338,250,250,3))
#for each dog breed (grouped by directory)
for oneDir in dirs:
    #get all the image name
    imageNames = (os.listdir(oneDir))
    #loop through each image in file
    for imgName in imageNames:
        #grab the image
        img =  PIL.Image.open(oneDir + imgName)
        #get the bounding box annotation
        f = open('root_data/Annotation/' + oneDir[17:] +imgName[:-4], "r")
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
        #resize the image
        resized_img = cropped_img.resize((250, 250))

        resized_img.show()
        #show the image

        image_array = np.array(resized_img)
        norm_image = image_array/255.0
        print(norm_image.shape)
        print(norm_image)

        image_array = np.array(resized_img)
        norm_image = np.clip(image_array/255.0, 0.0, 1.0) # 255 = max of the value of a pixel
        norm_image.show()

        
    i+=1
#https://medium.com/tensorflow/hello-deep-learning-fashion-mnist-with-keras-50fcff8cd74a
#this section needs to be recoded
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        #normalize the image
        image_array = np.array(resized_img)
        norm_image = image_array/255.0 # 255 = max of the value of a pixel
        dogPictureMatrix[i] = norm_image
        i+=1
print(dogPictureMatrix.shape)
X = dogPictureMatrix
#generate the target vector
y = np.zeros(338)
y[int(len(y)/2):]=1
print(y.shape)
#split the data into train/test/validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
print(X_train.shape)
#https://medium.com/tensorflow/hello-deep-learning-fashion-mnist-with-keras-50fcff8cd74
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(250, 250,3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)
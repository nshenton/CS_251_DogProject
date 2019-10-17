# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:57:15 2019

@author: luket
"""
#https://www.tensorflow.org/tutorials/keras/basic_classification
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
import os
import PIL
# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot
#root directory
rootDir = "root_data/images/n02098286-West_Highland_white_terrier/"
#list of image names
imageNames = (os.listdir(rootDir))
#dont blow up my computer
i = 0
#loop through each image in file
for imgName in imageNames:
    #for testing
    if i < 1:
        #grab the image
        img =  PIL.Image.open(rootDir + imgName)
        #get the bounding box annotation
        f = open('root_data/Annotation/n02098286-West_Highland_white_terrier/'+imgName[:-4], "r")
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
        #show the image
<<<<<<< HEAD
        data = list(resized_img.getdata())
        data = np.array(data)/255.0
        
=======
        image_array = np.array(resized_img)
        norm_image = np.clip(image_array/255.0, 0.0, 1.0) # 255 = max of the value of a pixel
        norm_image.show()
>>>>>>> LT
        
    i+=1
#https://medium.com/tensorflow/hello-deep-learning-fashion-mnist-with-keras-50fcff8cd74a
#this section needs to be recoded
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#
#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#
#train_images = train_images / 255.0
#
#test_images = test_images / 255.0
#
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=(28, 28)),
#    keras.layers.Dense(128, activation=tf.nn.relu),
#    keras.layers.Dense(10, activation=tf.nn.softmax)
#])
#
#model.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])
#
#model.fit(train_images, train_labels, epochs=5)
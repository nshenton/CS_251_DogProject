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
# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot
# load image as pixel array
data = image.imread('root_data/images/n02085620-Chihuahua/n02085620_7.jpg')
f = open('root_data/Annotation/n02085620-Chihuahua/n02085620_7', "r")
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
print(xmin,xmax,ymin,ymax)
f.close()
# load the image
print(os.listdir())
# summarize shape of the pixel array
print(data.dtype)
print(data.shape)
# display the array of pixels as an image
pyplot.imshow(data)
pyplot.show()
data = data[ymin:ymax, xmin:xmax, :]
pyplot.imshow(data)
pyplot.show()
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
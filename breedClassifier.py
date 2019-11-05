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
#how many breeds to build into the classifier
def pickleData(maxBreeds):
    #generate directory names for specified number of breeds
    dirs = os.listdir("root_data/images/")[:maxBreeds]
    #initialize the x dimension of the data matrix
    matrixXDim = 0 
    #initialize an index for which directory we are currently using
    currentBreed = 0
    #for each directory
    for oneDir in dirs:
        #count the number of images for this breed
        oneDir = "root_data/images/"+str(oneDir)
        imageNames = (os.listdir(oneDir))
        numImages = len(imageNames)
        #increment the x dimension of the data matrix
        matrixXDim+=numImages
        #maximum breeds for development purposes
        currentBreed+=1
        if currentBreed > maxBreeds:
            break
    #initialize the target matrix
    y = np.zeros(matrixXDim)
    #track the index of the target matrix for proper initialization
    prevIndex = 0
    #the number of the actual target (1,2,3,...)
    targetCounter = 0
    #for each directory of images
    for oneDir in dirs:
        #count the number of images
        oneDir = "root_data/images/"+str(oneDir)
        imageNames = (os.listdir(oneDir))
        numImages = len(imageNames)
        #initialize the proper number of indices to this breed in the target vector
        y[prevIndex:(numImages+prevIndex)] = targetCounter
        #increment the target vector
        targetCounter+=1
        #increment the index to start for the following breed
        prevIndex = (numImages+prevIndex)
    #initialize data matrix
    dogPictureMatrix = np.zeros((matrixXDim,250,250,3))
    #data martix index
    i = 0
    #max breed counter index
    currentBreed = 0
    #for each dog breed (grouped by directory)
    for oneDir in dirs:
        #adjust the actual folder name to index 
        oneDir = "root_data/images/"+str(oneDir)
        croppedDir = "root_data/imagescropped/"+str(oneDir)
        #get all the image names
        imageNames = (os.listdir(oneDir))
        #loop through each image
        for imgName in imageNames:
            #grab the image
            img =  PIL.Image.open(oneDir + "/" + imgName)
            #get the bounding box annotation
            f = open('root_data/Annotation/' + oneDir[17:] + "/"+imgName[:-4], "r")
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
            cropped_img.save(croppedDir + "/" + imgName)
            #resize the image
            #resized_img = cropped_img.resize((250, 250))
            #normalize the image
            #image_array = np.array(resized_img)
            #norm_image = image_array/255.0 # 255 = max of the value of a pixel
            #r = norm_image[:,:,0].reshape(1,62500)
            #g = norm_image[:,:,1].reshape(1,62500)
            #b = norm_image[:,:,2].reshape(1,62500)
            #data = pd.DataFrame(r.T,columns =['R'])
            #gDat = pd.DataFrame(g.T,columns =['R']) 
            #bDat = pd.DataFrame(b.T,columns =['B'])
            #frames = [rDat, gDat, bDat]
            #result = pd.concat(frames)
            #print(result)
            #break
            #pca = PCA(n_components = 2)
            #principalComponents = pca.fit_transform(rgb)
            #print(principalComponents.shape)
            #dogPictureMatrix[i] = principalComponents
            #i+=1
            currentBreed+=1
            if currentBreed > maxBreeds:
                #pickle.dump( dogPictureMatrix, open( "datamat.p", "wb" ) )
                #pickle.dump( y, open( "targetvec.p", "wb" ) )
                break
def splitAndTest(X, y, maxBreeds, save):
    #split the data into train/test/validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    #print the shape of the training data
    print(X_train.shape)
    #https://medium.com/tensorflow/hello-deep-learning-fashion-mnist-with-keras-50fcff8cd74
    #generate the model using keras
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(250, 250,3)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(maxBreeds, activation=tf.nn.softmax)
    ])
    #compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #fit the model
    results = model.fit(X_train, y_train, epochs=2)
    #get the accuracy of the last epoch
    print(results.history.get('acc')[-1])
    if save:
        #dump out the trained model
        model.save(str(maxBreeds) + "_breed_classifier.m")
totalBreeds = 2
pickleData(totalBreeds)
#declare X to be the data matrix
#data = pickle.load( open( "datamat.p", "rb" ) )
#target = pickle.load( open( "targetvec.p", "rb" ) )
#splitAndTest(data,target, totalBreeds, False)
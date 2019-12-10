#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 21:48:42 2019

@author: natshenton
"""

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
import glob
import random
from sklearn.model_selection import RandomizedSearchCV
from time import time
from scipy.stats import expon
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform
from scipy.stats import randint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.constraints import max_norm

def generateXy():
    #paths to images
    pure_img_paths = glob.glob("root_data/images/Images/*/*.jpg")
    mixed_img_paths = glob.glob("root_data/images/mixed/*")
    random.shuffle(pure_img_paths)
    random.shuffle(mixed_img_paths)
    print('Pure Data Set Size:', len(pure_img_paths))
    print('Mixed Data Set Size:', len(mixed_img_paths))
    #pretrained model
    model = ResNet50()
    #generate predictions
    arrayLen = (len(mixed_img_paths)*2)
    X = np.zeros((arrayLen,4))
    X = generatePredictions(X,pure_img_paths, True, arrayLen, model)
    X = generatePredictions(X,mixed_img_paths, False, arrayLen, model)
    print(X[0:2])
    #pca data matrix
    pcaComps = 2
    pca = PCA(n_components=pcaComps)
    #print(predictionVector.reshape(1, -1))
    X = pca.fit_transform(X)
    print(X[0:2])
    y = np.ones(arrayLen)
    y[int(arrayLen/2):arrayLen] = 0
    print(y)   
    return X,y

X,y = generateXy()

#%%
fig, ax = plt.subplots(figsize=(20,10))
idx = np.where(y == 1)
idx2 = np.where(y == 0)
ax.scatter(X[idx2 ,0], X[idx2,1], c='blue', edgecolor='k', s=20, label = 'mixed', alpha = 0.5)
ax.scatter(X[idx ,0], X[idx,1], c='red', edgecolor='k', s=20, label = 'pure', alpha = 0.5)
ax.set_xlabel(r'Pricipal Component 1', fontsize = 25)
ax.set_ylabel(r'Pricipal Component 2', fontsize = 25)
ax.set_title('Vizualizing the Data After PCA', fontsize = 20)
left, bottom, width, height = [0.2, 0.65, 0.2, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.scatter(X[idx2 ,0], X[idx2,1], c='blue', edgecolor='k', s=20, label = 'mixed', alpha = 0.5)
ax2.scatter(X[idx ,0], X[idx,1], c='red', edgecolor='k', s=20, label = 'pure', alpha = 0.5)
ax2.set_xlim(-0.35,-0.25)
ax2.set_ylim(-0.05,0.01)
ax.legend(fontsize = 20)
plt.savefig('Data_vis.png')
plt.show()

left, bottom, width, height = [0.2, 0.65, 0.2, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.scatter(X[idx2 ,0], X[idx2,1], c='blue', edgecolor='k', s=20, label = 'mixed', alpha = 0.5)
ax2.scatter(X[idx ,0], X[idx,1], c='red', edgecolor='k', s=20, label = 'pure', alpha = 0.5)
ax2.set_xlim(-0.35,-0.25)
ax2.set_ylim(-0.05,0.01)
ax.legend()
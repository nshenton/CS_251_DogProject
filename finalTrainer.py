# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:42:42 2019

@author: luket
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
from matplotlib.colors import LinearSegmentedColormap
import pickle
# Utility function to report best scores
#takes params results (from random search cv) and # to report (default 1)
def report(results, n_top=1):
    bestMeanTest = 0.0
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        j = 0
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            if j == 0:
                bestMeanTest = results['mean_test_score'][candidate]
                j+=1
    return bestMeanTest
#returns the matrix to use in training models
#takes in the matrix X, the path to photos
#whether this is pure bred photos (first = true) or mixed (first = false)
#how many pictures do we have for mixed bred dogs (the limiting case)
#the model to use to generate predictions (ResNet)
def generatePredictions(outputArray, path, first, localLen, localModel):
    i = 0
    #for each image path
    for img_path in path:
        #load in the image
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        #process the image and decode the predictions into a vector
        preds = localModel.predict(x)
        listPreds = decode_predictions(preds, top=4)[0]
        predictionVector = np.array((listPreds[0][2],listPreds[1][2],listPreds[2][2],listPreds[3][2]))
        #insert into training matrix, first half is pure second half is mixed
        if not first:
            outputArray[i+int(localLen/2)] = predictionVector
        else:
            outputArray[i] = predictionVector
        i+=1
        if i == int(localLen/2):
            return outputArray
#this method generates X and y to test models
def generateXy():
    #paths to images
    pure_img_paths = glob.glob("C:/Users/luket/Desktop/CS_251_DogProject/root_data/images/Images/*/*.jpg")
    mixed_img_paths = glob.glob("C:/Users/luket/Desktop/CS_251_DogProject/root_data/images/mixed/*")
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
    return X,y,pca
X,y,pca = generateXy()
print(pca)
pickle.dump(pca, open( "pca.p", "wb" ) )
pca.transform(np.array([0.0,0.9,0.4,0.8]).reshape(-1,1))
# define the keras model
EPOCHS = 300
bestModel = Sequential()
bestModel.add(Dense(48, input_dim=2, activation='relu',kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))
bestModel.add(Dense(24, activation='relu',kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))
bestModel.add(Dense(4, activation='relu',kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))
bestModel.add(Dense(1, activation='sigmoid'))
bestModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=2)
history = bestModel.fit(X_train, y_train, epochs=EPOCHS, batch_size=500, verbose = 1)
bestModel.evaluate(X_test, y_test)

# Plot the decision boundary
#generate grid of points
# Create a figure instance
fig = plt.figure(figsize=(20,10))
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.xlabel('Principal Component 1',fontsize = 25)
plt.ylabel('Principal Component 2',fontsize = 25)
h=0.01
x_min, x_max = X[:,0].min()-0.01, X[:,0].max()+.01
y_min, y_max = X[:,1].min()-.01, X[:,1].max()+.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

#obtain predictions on grid
Z = bestModel.predict(np.c_[xx.ravel(), yy.ravel()]) 
Z = Z.reshape(xx.shape)

#plot
myColors = [ (0, 0, 1),(1, 0, 0)]  # R  -> B
cmap_name = 'my_list'
idx = np.where(y == 1)
idx2 = np.where(y == 0)
plt.scatter(X[idx ,0], X[idx,1], c='red', edgecolor='k', s=40, label = 'pure')
plt.scatter(X[idx2 ,0], X[idx2,1], c='blue', edgecolor='k', s=40, label = 'mixed')
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired)
plt.contourf(xx, yy, Z, colors=myColors,alpha=.1)
plt.title("Neural Network Decision Boundary",fontsize=50)
plt.legend(loc=3, prop={'size': 30},markerscale=2., scatterpoints=1)
left, bottom, width, height = [0.18, 0.66, 0.18, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.scatter(X[idx2 ,0], X[idx2,1], c='blue', edgecolor='k', s=20, label = 'mixed', alpha = 0.5)
ax2.scatter(X[idx ,0], X[idx,1], c='red', edgecolor='k', s=20, label = 'pure', alpha = 0.5)
ax2.set_xlim(-0.3169,-0.3)
ax2.set_ylim(-0.03,-0.02)
ax2.contourf(xx, yy, Z, colors=myColors,alpha=.1)
plt.savefig("decboundary.png",transparent=False,dpi=100,bbox_inches = "tight")
plt.show()
#bestModel.save("bestModel.h5")

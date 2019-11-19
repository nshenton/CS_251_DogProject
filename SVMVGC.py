# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:15:01 2019

@author: luket
"""
import IPython.display as display
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.keras.constraints import max_norm
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications import xception
from tensorflow.keras.applications import inception_v3
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score
import glob
import random
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from time import time
from scipy.stats import expon
import random
from sklearn.decomposition import PCA
# Utility function to report best scores
def report(results, n_top=1):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
def generatePredictions(outputArray, path, first, localLen):
    i = 0
    for img_path in path:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        listPreds = decode_predictions(preds, top=4)[0]
        predictionVector = np.array((listPreds[0][2],listPreds[1][2],listPreds[2][2],listPreds[3][2]))
        if not first:
            outputArray[i+int(localLen/2)] = predictionVector
        else:
            outputArray[i] = predictionVector
        i+=1
        if i == int(localLen/2):
            return outputArray
pure_img_paths = glob.glob("C:/Users/luket/Desktop/CS_251_DogProject/root_data/images/Images/*/*.jpg")
mixed_img_paths = glob.glob("C:/Users/luket/Desktop/CS_251_DogProject/root_data/images/mixed/*")
random.shuffle(pure_img_paths)
random.shuffle(mixed_img_paths)
print('Pure Data Set Size:', len(pure_img_paths))
print('Mixed Data Set Size:', len(mixed_img_paths))
model = ResNet50()
arrayLen = (len(mixed_img_paths)*2)
X = np.zeros((arrayLen,4))
X = generatePredictions(X,pure_img_paths, True, arrayLen)
X = generatePredictions(X,mixed_img_paths, False, arrayLen)
print(X[0:2])
pcaComps = 2
pca = PCA(n_components=pcaComps)
#print(predictionVector.reshape(1, -1))
X = pca.fit_transform(X)
print(X[0:2])
y = np.ones(arrayLen)
y[int(arrayLen/2):arrayLen] = 0
print(y)   
            
# specify parameters and distributions to sample from
param_dist = {"C": expon(scale=2000),
             "gamma": expon(scale=1),
             "kernel":['linear','sigmoid','poly','rbf'],
             "class_weight":["balanced",None]}

# run randomized search
n_iter_search = 500
model = svm.SVC()
random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=30, iid=False)

start = time()
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=5)
random_search.fit(X_train, y_train)

#report results
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

bestModel = svm.SVC(C = random_search.best_params_['C'],
                    class_weight = random_search.best_params_['class_weight'],
                    gamma = random_search.best_params_['gamma'],
                    kernel = random_search.best_params_['kernel'])
bestModel.fit(X_train,y_train)
# Plot the decision boundary
#Source: https://stackoverflow.com/questions/22294241/plotting-a-decision-boundary-separating-2-classes-using-matplotlibs-pyplot
h = .001  # step size in the mesh
# create a mesh to plot in
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
Z = bestModel.predict(np.c_[xx.ravel(), yy.ravel()])
print(Z)

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, cmap=plt.cm.flag)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=40, cmap=plt.cm.flag);
plt.show()
acc = metrics.accuracy_score(bestModel.predict(X_test),y_test)
print("Accuracy: " + str(acc))
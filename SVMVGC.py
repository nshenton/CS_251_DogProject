# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:15:01 2019

Script Function:
Generate vector of predictions from input photos
PCA vector to 2-dimensions
Split data into train and test set
Tune svm or random forest parameters using CV on train split
Report test accuracy

@author: luket
"""
from tensorflow.keras.applications.resnet50 import ResNet50
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
# Utility function to report best scores
#takes params results (from random search cv) and # to report (default 1)
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
    return X,y
#this class takes in the train and test data matrix and a param identifying what type of model to test on
def splitAndTest(X, y, svmMod):
    # specify parameters and distributions to sample from (SVM)
    param_dist_svm = {"C": expon(scale=1000),
                 "gamma": expon(scale=2),
                 "kernel":['poly','rbf'],
                 "class_weight":["balanced",None]}
    # specify parameters and distributions to sample from (random forest)
    param_dist_rf = {"min_samples_split": uniform(),
                     "max_depth":randint(1,5)}
    
    # run randomized search
    n_iter_search = 2000
    
    if svmMod: 
        model = svm.SVC()
        random_search = RandomizedSearchCV(model, param_distributions=param_dist_svm,
                                       n_iter=n_iter_search, cv=3, iid=False)
    else:   
        model = RandomForestClassifier(n_estimators=100)
        random_search = RandomizedSearchCV(model, param_distributions=param_dist_rf,
                               n_iter=n_iter_search, cv=3, iid=False)
    
    #tune params and report accuracy n times
    for i in range(5):
        start = time()
        #split
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=random.randint(0,10000))
        #fit on training set
        random_search.fit(X_train, y_train)
        #report results
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), n_iter_search))
        report(random_search.cv_results_)
        #tune parameters
        if svmMod:
            bestModel = svm.SVC(C = random_search.best_params_['C'],
                                class_weight = random_search.best_params_['class_weight'],
                                gamma = random_search.best_params_['gamma'],
                                kernel = random_search.best_params_['kernel'])
        else:
            bestModel = RandomForestClassifier(n_estimators=100, max_depth=random_search.best_params_['max_depth'], 
                                           min_samples_split = random_search.best_params_['min_samples_split'])
        bestModel.fit(X_train,y_train)
        # Plot the decision boundary for the test data
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
#X,y = generateXy()
splitAndTest(X,y, True)
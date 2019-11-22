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
    return X,y
#this class takes in the train and test data matrix and a param identifying what type of model to test on
def splitAndTest(X, y, modelToUse):
    # specify parameters and distributions to sample from (SVM)
    param_dist_svm = {"C": expon(scale=1000),
                 "gamma": expon(scale=2),
                 "kernel":['poly','rbf'],
                 "class_weight":["balanced",None]}
    
    param_dist_svm_linear = {"C": expon(scale=1000),
                 "gamma": expon(scale=2),
                 "kernel":['linear'],
                 "class_weight":["balanced",None]}
    # specify parameters and distributions to sample from (random forest)
    param_dist_rf = {"min_samples_split": uniform(),
                     "max_depth":randint(1,50)}
    
    # run randomized search
    n_iter_search = 200
    if modelToUse == -1: 
        model = svm.SVC()
        random_search = RandomizedSearchCV(model, param_distributions=param_dist_svm_linear,
                                       n_iter=n_iter_search, cv=15, iid=False)
    if modelToUse == 0: 
        model = svm.SVC()
        random_search = RandomizedSearchCV(model, param_distributions=param_dist_svm,
                                       n_iter=n_iter_search, cv=15, iid=False)
    elif modelToUse == 1: 
        model = RandomForestClassifier(n_estimators=30)
        random_search = RandomizedSearchCV(model, param_distributions=param_dist_rf,
                               n_iter=n_iter_search, cv=15, iid=False)
    elif modelToUse == 2:
        # define the keras model
        EPOCHS = 300
        bestModel = Sequential()
        bestModel.add(Dense(48, input_dim=2, activation='relu',kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))
        bestModel.add(Dense(24, activation='relu',kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))
        bestModel.add(Dense(4, activation='relu',kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))
        bestModel.add(Dense(1, activation='sigmoid'))
        bestModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #tune params and report accuracy n times
    avgOver = 50
    accuracyReport = np.zeros(avgOver)
    trainTestReport = np.zeros(avgOver)
    trainResults = 0.0
    for i in range(avgOver):
        #split
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=i)
        #tune parameters
        if modelToUse == 0 or modelToUse == -1:
            start = time()
            #fit on training set
            random_search.fit(X_train, y_train)
            #report results
            print("RandomizedSearchCV took %.2f seconds for %d candidates"
                  " parameter settings." % ((time() - start), n_iter_search))
            trainResults = report(random_search.cv_results_)
            bestModel = svm.SVC(C = random_search.best_params_['C'],
                                class_weight = random_search.best_params_['class_weight'],
                                gamma = random_search.best_params_['gamma'],
                                kernel = random_search.best_params_['kernel'])
            bestModel.fit(X_train,y_train)
        elif modelToUse == 1:
            start = time()
            #fit on training set
            random_search.fit(X_train, y_train)
            #report results
            print("RandomizedSearchCV took %.2f seconds for %d candidates"
                  " parameter settings." % ((time() - start), n_iter_search))
            trainResults = report(random_search.cv_results_)
            bestModel = RandomForestClassifier(n_estimators=10, max_depth=random_search.best_params_['max_depth'], 
                                           min_samples_split = random_search.best_params_['min_samples_split'])
            bestModel.fit(X_train,y_train)
        elif modelToUse == 2:
            history = bestModel.fit(X_train, y_train, epochs=EPOCHS, batch_size=500, verbose = 0)
            trainResults = (history.history['acc'][-1])
        # Plot the decision boundary for the test data
        #Source: https://stackoverflow.com/questions/22294241/plotting-a-decision-boundary-separating-2-classes-using-matplotlibs-pyplot
        plot = False
        if plot:
            h = .001  # step size in the mesh
            # create a mesh to plot in
            x_min, x_max = X[:, 0].min(), X[:, 0].max()
            y_min, y_max = X[:, 1].min(), X[:, 1].max()
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            
            
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
            Z = bestModel.predict(np.c_[xx.ravel(), yy.ravel()])
            
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, cmap=plt.cm.flag)
            plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=40, cmap=plt.cm.flag);
            plt.show()
        if modelToUse == 2:
            indH, accuracy = bestModel.evaluate(X_test, y_test)
            print('Accuracy: %.2f' % (accuracy*100))
            #performance metrics
            accuracyReport[i] = accuracy
            trainTestReport[i] = (accuracy -trainResults)
            if plot:
                acc = history.history['acc']
                loss = history.history['loss']
                
                epochs_range = range(EPOCHS)
                
                plt.figure(figsize=(8, 8))
                plt.subplot(1, 2, 1)
                plt.plot(epochs_range, acc, label='Training Accuracy')
                plt.legend(loc='lower right')
                plt.title('Training Accuracy')
                
                plt.subplot(1, 2, 2)
                plt.plot(epochs_range, loss, label='Training Loss')
                plt.legend(loc='upper right')
                plt.title('Training Loss')
                plt.show()
        else:
            acc = metrics.accuracy_score(bestModel.predict(X_test),y_test)
            print("Accuracy: " + str(acc))
            accuracyReport[i] = acc
            trainTestReport[i] = (acc -trainResults)
    print(accuracyReport)
    print(trainTestReport)
    return [accuracyReport,trainTestReport]
#X,y = generateXy()
start2 = time()
accuracyArray = list()
trainTestArray = list()
returnArray = splitAndTest(X,y,-1)
accuracyArray.append(returnArray[0])
trainTestArray.append(returnArray[1])
returnArray = splitAndTest(X,y,0)
accuracyArray.append(returnArray[0])
trainTestArray.append(returnArray[1])
returnArray = splitAndTest(X,y,1)
accuracyArray.append(returnArray[0])
trainTestArray.append(returnArray[1])
returnArray = splitAndTest(X,y,2)
accuracyArray.append(returnArray[0])
trainTestArray.append(returnArray[1])
# Create a figure instance
fig = plt.figure(figsize=(10,5))

# Create an axes instance
ax = fig.add_axes([0,0,1,1])
# Create the boxplot
bp = ax.violinplot(accuracyArray)
plt.setp(ax, xticks=[y + 1 for y in range(len(accuracyArray))],
         xticklabels=['SVM (linear kernel)', 'SVM (optimal kernel)', 'Random Forest', 'Neural Network'])
plt.title('Accuracy distribution by model type', fontsize = 18)
ax.tick_params(labelsize=13)
plt.savefig("testOut.png",transparent=False,dpi=100,bbox_inches = "tight")
        
plt.show()

# Create a figure instance
fig = plt.figure(figsize=(10,5))

# Create an axes instance
ax = fig.add_axes([0,0,1,1])
# Create the boxplot
bp = ax.violinplot(trainTestArray)
plt.setp(ax, xticks=[y + 1 for y in range(len(accuracyArray))],
         xticklabels=['SVM (linear kernel)', 'SVM (optimal kernel)', 'Random Forest', 'Neural Network'])
plt.title('Test-train accuracy distribution by model type', fontsize = 18)
ax.tick_params(labelsize=13)
plt.savefig("testOut2.png",transparent=False,dpi=100,bbox_inches = "tight")
        
plt.show()
print(str((time() - start2)))
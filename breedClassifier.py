# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:32:16 2019

@author: luket
"""
import IPython.display as display
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from keras.preprocessing.image import ImageDataGenerator
#small code chunk to split folders into train/val/test
#import split_folders
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
#split_folders.ratio('C:/Users/luket/Desktop/CS_251_DogProject/root_data/Images/Images', output="output", seed=1337, ratio=(.8, .1, .1)) # default values

#create image data generators for training and test sets
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        "C:/Users/luket/Desktop/CS_251_DogProject/root_data/images/train",
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        "C:/Users/luket/Desktop/CS_251_DogProject/root_data/images/val",
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

# define the keras model
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(150, 150 ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(120, activation='sigmoid')
])

#compile the model
model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

#fit the model
model.fit_generator(
        train_generator,
        steps_per_epoch=500,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=200)
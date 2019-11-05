# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:13:49 2019

@author: luket
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import pathlib
import numpy as np
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#end import section
#not sure what this is doing, see tutorial: https://www.tensorflow.org/tutorials/load_data/images
AUTOTUNE = tf.data.experimental.AUTOTUNE
#source of the images locally
data_dir = pathlib.Path("C:/Users/luket/Desktop/CS_251_DogProject/root_data/Images")
#how many images do we have
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
#grab class names
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
print(CLASS_NAMES)
#show a few photos
#mexicanHairless = list(data_dir.glob('n02113978-Mexican_hairless/*'))
#for image_path in mexicanHairless[:3]:
#    display.display(Image.open(str(image_path)))
#simple way to load images
# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#define parameters
BATCH_SIZE = 32
IMG_HEIGHT = 250
IMG_WIDTH = 250
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
#training data generator
train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')
image_batch, label_batch = next(train_data_gen)
show_batch(image_batch, label_batch)
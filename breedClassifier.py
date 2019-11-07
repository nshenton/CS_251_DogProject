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
from sklearn.metrics import confusion_matrix
#small code chunk to split folders into train/val/test
#import split_folders
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
#split_folders.ratio('C:/Users/luket/Desktop/CS_251_DogProject/root_data/Images/Images', output="output", seed=1337, ratio=(.8, .1, .1)) # default values

#create image data generators for training and test sets
shift = 0.2
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.5,
        zoom_range=0.5,
        horizontal_flip=True,
        rotation_range = 90)

val_datagen = ImageDataGenerator(rescale=1./255,
        shear_range=0.5,
        zoom_range=0.5,
        horizontal_flip=True,
        rotation_range = 90)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        "C:/Users/luket/Desktop/CS_251_DogProject/root_data/images/train",
        target_size=(250, 250),
        batch_size=32,
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        "C:/Users/luket/Desktop/CS_251_DogProject/root_data/images/val",
        target_size=(250, 250),
        batch_size=32,
        class_mode='categorical')

# define the keras model
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(250, 250 ,3)),
    MaxPooling2D(),
    Dropout(0.4),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.4),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(120, activation='sigmoid')
])

#compile the model
model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

#fit the model
EPOCHS = 1
history = model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=5)

#performance metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#test metrics
test_generator = test_datagen.flow_from_directory(
        "C:/Users/luket/Desktop/CS_251_DogProject/root_data/images/test",
        target_size=(250, 250),
        batch_size=5,
        class_mode='categorical',
        shuffle = False)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)
print(predicted_class_indices)

#labels = (train_generator.class_indices)
#labels = dict((v,k) for k,v in labels.items())
#predictions = [labels[k] for k in predicted_class_indices]

#confusion_matrix(predicted_class_indices,predictions)
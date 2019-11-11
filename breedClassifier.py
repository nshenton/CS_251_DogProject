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
from tensorflow.keras.constraints import max_norm
from sklearn.utils.multiclass import unique_labels
#small code chunk to split folders into train/val/test
#import split_folders
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
#split_folders.ratio('C:/Users/luket/Desktop/CS_251_DogProject/root_data/Images/Images', output="output", seed=1337, ratio=(.8, .1, .1)) # default values

#create image data generators for training and test sets
shift = 0.2
IMAGE_SIZE = 150
BATCH_SIZE = 300
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.8,
        zoom_range=0.8,
        horizontal_flip=True,
        rotation_range = 90)

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        "C:/Users/luket/Desktop/CS_251_DogProject/root_data/images/train",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        "C:/Users/luket/Desktop/CS_251_DogProject/root_data/images/val",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

# define the keras model
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE ,3)),
    MaxPooling2D(),
    Dropout(0.5),
    Conv2D(32, 3, padding='same', activation='relu',kernel_constraint=max_norm(3), bias_constraint=max_norm(3)),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu',kernel_constraint=max_norm(3), bias_constraint=max_norm(3)),
    MaxPooling2D(),
    Dropout(0.5),
    Flatten(),
    Dense(512, activation='relu',kernel_constraint=max_norm(3), bias_constraint=max_norm(3)),
    Dense(120, activation='sigmoid')
])

#compile the model
model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

#fit the model
EPOCHS = 2
history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n//train_generator.batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.n//validation_generator.batch_size)

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
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=1,
        class_mode='categorical',
        shuffle = False)

test_generator.reset()
pred=model.predict_generator(test_generator,
steps=test_generator.n,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)
print(predicted_class_indices)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           #xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plot_confusion_matrix(test_generator.classes, predicted_class_indices, classes=test_generator.classes, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
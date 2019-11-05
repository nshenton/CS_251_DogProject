# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:13:49 2019

@author: luket
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import pathlib
AUTOTUNE = tf.data.experimental.AUTOTUNE
data_dir = pathlib.Path("C:/Users/luket/Desktop/CS_251_DogProject/root_data/Images")
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import adagrad, adadelta, rmsprop, adam
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
from sklearn.cross_validation import StratifiedKFold
import numpy as np
import pandas as pd
import random
import math
import cv2


if __name__ == '__main__':
    colormode = 'rgb'
    channels = 3
    batchsize = 50
    trainingsamples = 4000
    model_name = 'thumbnails_27_inception_2'
    model = load_model('D:\\models\\slide_level\\thumbnails_27_inception_1.h5')

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=keras.optimizers.Adadelta(lr='0.1'), loss='categorical_crossentropy',  metrics=['accuracy'])
    train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
            "E:\\data_thumbnails\\train",
            target_size=(150, 150),
            batch_size=batchsize,
            color_mode=colormode)

    validation_generator = val_datagen.flow_from_directory(
            "E:\\data_thumbnails\\val",
            target_size=(150, 150),
            batch_size=batchsize,
            color_mode=colormode)

    history = model.fit_generator(
            train_generator,
            samples_per_epoch=trainingsamples,
            nb_epoch=100,
            validation_data=validation_generator,
            nb_val_samples=700)

    hist = history.history
    hist = pd.DataFrame(hist)
    hist.to_csv('D:\\results\\slide_level\\'+model_name+'.csv')
    model.save('D:\\models\\slide_level\\'+model_name+'.h5')
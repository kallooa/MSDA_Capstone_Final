from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import adagrad, adadelta, rmsprop, adam
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
from sklearn.cross_validation import StratifiedKFold
import numpy as np
import pandas as pd
import random
import math
import cv2

def create_model(channels, num_conv_filters_layer1, num_conv_kernel_rows, num_conv_kernel_cols, num_conv_filters_layer2):
    model = Sequential()
    act = 'relu'
    nb_classes = 27

    model.add(Convolution2D(num_conv_filters_layer1, num_conv_kernel_rows, num_conv_kernel_cols, border_mode='same', input_shape=(channels, 150, 150)))
    model.add(Activation(act))
    model.add(Convolution2D(num_conv_filters_layer1, num_conv_kernel_rows, num_conv_kernel_cols))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(num_conv_filters_layer2, num_conv_kernel_rows, num_conv_kernel_cols, border_mode='same'))
    model.add(Activation(act))
    model.add(Convolution2D(num_conv_filters_layer2, num_conv_kernel_rows, num_conv_kernel_cols))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(num_conv_filters_layer2, num_conv_kernel_rows, num_conv_kernel_cols, border_mode='same'))
    model.add(Activation(act))
    model.add(Convolution2D(num_conv_filters_layer2, num_conv_kernel_rows, num_conv_kernel_cols))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(num_conv_filters_layer2, num_conv_kernel_rows, num_conv_kernel_cols, border_mode='same'))
    model.add(Activation(act))
    model.add(Convolution2D(num_conv_filters_layer2, num_conv_kernel_rows, num_conv_kernel_cols))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(num_conv_filters_layer2, num_conv_kernel_rows, num_conv_kernel_cols, border_mode='same'))
    model.add(Activation(act))
    model.add(Convolution2D(num_conv_filters_layer2, num_conv_kernel_rows, num_conv_kernel_cols))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation(act))
    model.add(Dense(128))
    model.add(Activation(act))
    model.add(Dense(128))
    model.add(Activation(act))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model

if __name__ == '__main__':
    colormode = 'rgb'
    channels = 3
    batchsize = 25
    trainingsamples = 20000
    model_name = '27_class_cnn_color_1'

    model = create_model(channels, 48, 3, 3, 32)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
            "G:\\data\\train",
            target_size=(150, 150),
            batch_size=batchsize,
            color_mode=colormode)

    validation_generator = test_datagen.flow_from_directory(
            "G:\\data\\val",
            target_size=(150, 150),
            batch_size=batchsize,
            color_mode=colormode)

    history = model.fit_generator(
            train_generator,
            samples_per_epoch=trainingsamples,
            nb_epoch=100,
            validation_data=validation_generator,
            nb_val_samples=10000)

    hist = history.history
    hist = pd.DataFrame(hist)
    hist.to_csv('D:\\results\\'+model_name+'.csv')
    model.save('D:\\models\\'+model_name+'.h5')
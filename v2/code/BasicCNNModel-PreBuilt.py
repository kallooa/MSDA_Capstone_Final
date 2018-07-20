# Not yet tested

#Import Libraries:
from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import adam
from keras.models import Model
from keras.applications.vgg19 import VGG19


import numpy as np
import pandas as pd
import random
import math

if __name__ == '__main__':
    colormode = 'rgb'
    channels = 3 #color images have 3 channels. grayscale images have 1 channel
    batchsize = 1 #Number of images to be used in each processing batch. Larger batches have a greater impact on training accuracy but that isn't always a good thing
    trainingsamples = 25 #Number of images to be used for training set
    validationsamples = 25 #Number of images to be used for validation set
    model_name = 'KovalModel2' #Any name for saving and keeping track of this model
    numclasses = 2
    root_dir = 'C:\\Users\\Aadi\\Documents\\GitHub\\KovalCNN\\'
    
    # create the base pre-trained model
    base_model = VGG19(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(numclasses, activation='softmax')(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = True

    for layer in model.layers:
        layer.trainable = True
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='binary_crossentropy',  metrics=['accuracy']) #create model with for binary output with the adam optimization algorithm
    
    train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True) # use ImageDataGenerator to enhance the size of our dataset by randomly flipping images. There are many more transformations that are possible
    test_datagen = ImageDataGenerator()

#the following code reads images, trains the model, and saves the training history to a csv file:

    train_generator = train_datagen.flow_from_directory(
            root_dir+"data\\train",
            target_size=(150, 150),
            batch_size=batchsize,
            color_mode=colormode)

    validation_generator = test_datagen.flow_from_directory(
            root_dir+"data\\val",
            target_size=(150, 150),
            batch_size=batchsize,
            color_mode=colormode)

    history = model.fit_generator(
            train_generator,
            steps_per_epoch=trainingsamples/batchsize,
            epochs=100,
            validation_data=validation_generator,
            validation_steps=validationsamples/batchsize)

    hist = history.history
    hist = pd.DataFrame(hist)
    hist.to_csv(root_dir+'results\\'+model_name+'.csv')
    model.save(root_dir+'models\\'+model_name+'.h5')

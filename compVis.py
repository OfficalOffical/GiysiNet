
import cv2
import keras
from keras.applications.resnet_v2 import ResNet50V2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import GlobalAveragePooling2D

import editCsv

from keras.datasets import mnist




width = 128
height = 128
nRowSetter = 1000



def getImageFromCSV(csvPath):
    csv,imarr = editCsv.getAndCleanCsv(csvPath,width,height,nRowSetter)
    modelKeras = createKerasModel()






    trainX, testX, trainY, testY = train_test_split(imarr,csv["categoryx_id"],test_size=0.3,random_state=42)
    # u can devide it to 255 to make it 0-1 in further i guess dk


    print(modelKeras.summary())



    print(trainX.shape,testX.shape,trainY.shape,testY.shape)


    modelKeras.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['sparse_categorical_accuracy'])

    modelKeras.fit(trainX,trainY,epochs=15)
    modelKeras.evaluate(testX,testY)


    """  
    mnist %99 acc
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    print(modelKeras.summary())


    print(train_X.shape,train_y.shape,test_X.shape,test_y.shape)


    modelKeras.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['sparse_categorical_accuracy'])

    modelKeras.fit(train_X,train_y,epochs=15)
    modelKeras.evaluate(test_X,test_y)"""


    return csv

def createKerasModel():
    # Input Shape
    base_model = ResNet50V2(weights='imagenet',
                            include_top=False,
                            input_shape=(width, height, 3))

    base_model.trainable = True

    model = Sequential([
        base_model,

        Flatten(),  # Is it mandatory ? end to end conv2d ?

        Dense(164, activation="softmax"),
        # 1000 -> loss 3.85 \ acc 0.09
    ])


    """ model = Sequential([
    #128x128 acc %22
        keras.Input(shape=(width,height,3)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(2048, (8, 8), activation="relu"),

        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(1024, (6, 6), activation='relu'),

        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(512, (4, 4), activation='relu'),

        Conv2D(256, (3, 3), activation='relu'),

        Flatten(), # Is it mandatory ? end to end conv2d ?

        Dense(164, activation="softmax"),
        #1000 -> loss 3.85 \ acc 0.09
    ])"""






    return model




















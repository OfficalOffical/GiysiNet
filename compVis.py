
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




width = 64
height = 64
nRowSetter = 4000



def getImageFromCSV(csvPath):
    csv,imarr = editCsv.getAndCleanCsv(csvPath,width,height,nRowSetter)
    modelKeras = createKerasModel()



    print(imarr.shape)


    trainX, testX, trainY, testY = train_test_split(imarr,csv["categoryx_id"],test_size=0.3,random_state=42)
    # u can devide it to 255 to make it 0-1 in further i guess dk


    print(modelKeras.summary())



    print(trainX.shape,testX.shape,trainY.shape,testY.shape)


    modelKeras.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['sparse_categorical_accuracy'])

    modelKeras.fit(trainX,trainY,epochs=15)
    modelKeras.evaluate(testX,testY)


    return csv

def createKerasModel():
    # Input Shape
    base_model = ResNet50V2(weights='imagenet',
                            include_top=False,
                            input_shape=(width, height, 3))

    base_model.trainable = False

    model = Sequential([
        keras.Input(shape=(width,height,3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(2048, (8, 8), activation="relu"),
        Dropout(0.2),

        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(1024, (6, 6), activation='relu'),

        Conv2D(512, (4, 4), activation='relu'),

        Conv2D(256, (3, 3), activation='relu'),

        Conv2D(128, (2, 2), activation='relu'),

        Dense(164, activation="softmax"),

    ])
    model2 = Sequential([
        base_model,

        Conv2D(1024, (2, 2), activation='relu'),

        Dense(5, activation="softmax"),
    ])

    """
          Conv2D(1024, (2, 2), activation='relu'),

          Conv2D(512, (1, 1), activation='relu'),

          Conv2D(256, (1, 1), activation='relu'),

          """



    """ 
         Conv2D(2048, (1, 1), activation="relu"),
         Dropout(0.2),
         Conv2D(1024, (1, 1), activation='relu'),
         Dropout(0.2),
         Conv2D(512,  (1, 1), activation='relu'),
         Dropout(0.2),
         Conv2D(216,  (1, 1), activation='relu'),
    """

    return model




















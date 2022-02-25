
import cv2
from keras.applications.resnet_v2 import ResNet50V2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import GlobalAveragePooling2D

import editCsv




width = 128
height = 128
nRowSetter = 1000



def getImageFromCSV(csvPath):
    csv,imarr = editCsv.getAndCleanCsv(csvPath,width,height,nRowSetter)
    modelKeras = createKerasModel()






    trainX, testX, trainY, testY = train_test_split(imarr,csv["categoryx_id"],test_size=0.3,random_state=42)
    # u can devide it to 255 to make it 0-1 in further i guess dk


    print(modelKeras.summary())

    plt.figure(figsize=(7,20))




    modelKeras.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

    modelKeras.fit(trainX,trainY,epochs=10)
    modelKeras.evaluate(testX,testY)
    print("TrainX : ",trainX.shape)

    return csv

def createKerasModel():
    # Input Shape
    base_model = ResNet50V2(weights='imagenet',
                            include_top=False,
                            input_shape=(width, height, 3))

    base_model.trainable = False

    model = Sequential([
        base_model,
        Conv2D(128, (2, 2), activation="relu"),
        Conv2D(64, (2, 2), activation="relu"),
        Conv2D(32, (2, 2), activation="relu"),
        Dense(164, activation="softmax")
    ])



    return model




















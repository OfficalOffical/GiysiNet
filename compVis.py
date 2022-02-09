import glob
import urllib.request

import numpy as np
import os
import cv2
import tensorflow as tf
import keras
from keras import Model
import requests
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D, Dense
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import pandas as pd
from sklearn.model_selection import train_test_split

def getImageFromDest(datasetPath,csv):

    tempImage = []

    w = 32
    h = 32
    resizeRate = (w, h)

    for i in range(len(csv)):
            file = datasetPath+"/"+str(csv['set_id'][i]) + "/" + str(csv['index'][i])+".jpg"
            tempImage.append(cv2.resize(cv2.imread(file), resizeRate))


    image = np.array(tempImage)
    return image





def getImageFromCSV(csvPath):
    csv,imarr = getCsv(csvPath)
    modelKeras = createKerasModel()

    trainX, testX, trainY, testY = train_test_split(imarr,csv["categoryx_id"],test_size=0.3)
    # u can devide it to 255 to make it 0-1 in further i guess dk

    modelKeras.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


    modelKeras.fit(trainX,trainY,epochs=3)
    modelKeras.evaluate(testX,testY)



    return csv

def createKerasModel():
    # Input Shape
    img_width, img_height, _ = 32, 32, 3

    # Pre-Trained Model
    base_model = ResNet50(weights='imagenet',
                          include_top=False,
                          input_shape=(img_width, img_height, 3))
    base_model.trainable = False

    # Add Layer Embedding
    model = keras.Sequential([
        base_model,


    ])
    model.add(Dense(4526, activation='softmax'))


    return model


def getCsv(csvPath):
    datasetPath = "E:/db"
    temp = pd.read_csv(csvPath,nrows=10000)  # <<< burayı düzelt
    temp = temp.drop(labels=["id","color", "sub_category", "name", "all"], axis=1)
    temp = temp.sort_values(by=['set_id', 'index'])
    imageArray = getImageFromDest(datasetPath,temp)

    return temp,imageArray


















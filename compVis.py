
import cv2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import editCsv

width = 28
height = 28

def getImageFromDest(datasetPath,csv):

    tempImage = []

    resizeRate = (width, height)

    for i in range(len(csv)):
            file = datasetPath+"/"+str(csv['set_id'][i]) + "/" + str(csv['index'][i])+".jpg"
            tempImage.append(cv2.resize(cv2.imread(file), resizeRate))


    image = np.array(tempImage)
    return image





def getImageFromCSV(csvPath):
    csv,imarr = getCsv(csvPath)
    modelKeras = createKerasModel()

    print(csv["categoryx_id"])

    trainX, testX, trainY, testY = train_test_split(imarr,csv["categoryx_id"],test_size=0.3,random_state=42)
    # u can devide it to 255 to make it 0-1 in further i guess dk

    print(editCsv.labelNormalizer(trainY))



    """
    print(trainX.shape)
    print("A")

    modelKeras.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

    modelKeras.fit(trainX,trainY,epochs=3)
    modelKeras.evaluate(testX,testY)

    """

    return csv

def createKerasModel():
    # Input Shape
    model = Sequential([
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(5000, activation='softmax')

    ])




    return model


def getCsv(csvPath):
    datasetPath = "E:/db"
    temp = pd.read_csv(csvPath,nrows=10000)  # <<< burayı düzelt
    temp = temp.drop(labels=["id","color", "sub_category", "name", "all"], axis=1)
    temp = temp.sort_values(by=['set_id', 'index'])
    imageArray = getImageFromDest(datasetPath,temp)

    return temp,imageArray


















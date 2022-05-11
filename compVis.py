import random

import cv2
import keras
import matplotlib.pyplot as plt
import numpy
from keras.applications.resnet_v2 import ResNet50V2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.optimizers import adam_v2
import pandas as pd
import editCsv
import numpy as np
from scipy.spatial import distance
from keras import backend as K



width = 128
height = 128
nRowSetter = 1000



def getImageFromCSV(csvPath):
    csv,imarr = editCsv.getAndCleanCsv(csvPath,width,height,nRowSetter)
    modelKeras = createKerasModel()

    plt.figure(figsize=(7, 25))
    csv.categoryx_id.value_counts().sort_values().plot(kind='barh')
    plt.show()

    trainX, testX, trainY, testY = train_test_split(imarr,csv["categoryx_id"],test_size=0.3,random_state=42)
    # u can devide it to 255 to make it 0-1 in further i guess dk







    print(modelKeras.summary())



    print(trainX.shape,testX.shape,trainY.shape,testY.shape)

    #0.001 def
    opt = adam_v2.Adam(learning_rate=0.0001)

    modelKeras.compile(loss='sparse_categorical_crossentropy',optimizer= opt ,metrics=['sparse_categorical_accuracy'])


    modelKeras.fit(trainX,trainY,epochs=10) # batch size 64 ?

    evScore= modelKeras.evaluate(testX, testY)



    pp = modelKeras.predict(testX)










    #can be lower the features with PCA i guess


    whichPhoto = random.randint(1,((int(nRowSetter*0.3)-1)))

    simPic = [ distance.cosine(pp[whichPhoto],img )for img in pp] # 25. fotoyla cosine sim yapıyor




    bipbop = sorted(range(len(simPic)),key=lambda k:simPic[k])




    #bipbop = createBipBop(fullbop,testY)

    #bipbop = sorted(range(len(simPic)), key=lambda k: simPic[k])[1:6] <<



    firstCon = cv2.hconcat([testX[whichPhoto], testX[bipbop[0]]])
    secCon = cv2.hconcat([testX[bipbop[1]], testX[bipbop[2]]])
    thrdCon = cv2.hconcat([testX[bipbop[3]], testX[bipbop[4]]])

    sumCon = cv2.vconcat([firstCon, secCon])
    sumCon2 = cv2.vconcat([sumCon, thrdCon])

    plt.figure(figsize=(32, 4))
    plt.plot(simPic)
    plt .show()


    plt.figure(figsize=(16,4))
    plt.plot(pp[25])
    plt.show()


    cv2.imshow("A", sumCon2)
    cv2.waitKey()




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
        MaxPooling2D(pool_size=(2, 2)),
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




def createBipBop(fullbop,testY):



    bipbop = [0,0,0,0,0]
    maxAcc = [0,0,0,0,0]
    for x in range(len(fullbop)):
        classTemp = belongedClass(x)
        if( fullbop[x] >= maxAcc[classTemp]):
            maxAcc[classTemp] = fullbop[x]








    pass

def belongedClass(temp):

    print("AAAAAAAA")
    print(temp)

    """ # 0 clothing şunba bi bak
    23 spor giyim
    man clothing 85
    teapods 126
    juniors 127
    129
    145 Boys
    146 Girls
    """

    accesories = [24,25,26,27,28,29,38,39,40,41,42,43,44,45,46,47,51,73,74,75,84,103,112,113,114,115,116,117,118,122,123,124,128,137,138,139,142,143,144,154,155,157,158,159]
    top = [9,11,12,13,14,48,60,62,63,67,72,86,87,88,99,120,125,130,133,147,148,149,150,152,163]
    overLayer = [10,15,16,17,52,59,71,89,90,97,102,131,132]
    fullBody = [1,2,3,21,22,58,64,65,94,95,98,151,160,161,]
    down = [4,5,6,7,8,18,19,20,53,54,55,56,57,61,66,68,69,70,91,92,93,96,100,101,119,121,129,134,135,153,162]
    shoes = [30,31,32,33,34,35,36,37,49,50,76,77,78,79,80,81,82,83,104,105,106,107,108,109,110,111,136,140,141,156]


    return temp















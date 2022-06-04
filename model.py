import gc
from random import randint

import cv2
import numpy as np
from keras import Sequential
from keras import backend as K
from keras.applications.resnet_v2 import ResNet101V2
from keras.applications.vgg19 import VGG19
from keras.layers import Flatten, Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, ZeroPadding2D, Dropout
from keras.models import load_model
from scipy.spatial import distance
from sklearn.model_selection import train_test_split

width = 224
height = 224
epochNumber = 40
nRowSetter = 2000

whichPhoto = randint(1, (int(nRowSetter * 0.3) - 1))


def mainModel(csv, img, mode):

    cv2.imshow("template", cv2.imread("Template.png"))


    trainX, testX, trainY, testY = train_test_split(img, csv, test_size=0.3, random_state=42)

    if mode == 3:
        evScoreVGG, imHolderVGG, imHolderVGGEarly = runKerasVGGModel(trainX, testX, trainY, testY, mode)
        print("VGG        |", evScoreVGG[1])
        cv2.imshow("VGG", imHolderVGG)


    else:

        evScoreVGG, imHolderVGG, imHolderVGGEarly = runKerasVGGModel(trainX, testX, trainY, testY, mode)

        print("VGG        |", evScoreVGG[1])
        cv2.imshow("VGG", imHolderVGG)
        cv2.imshow("VGGEarly", imHolderVGGEarly)
        del (evScoreVGG, imHolderVGG)
        gc.collect()

        cv2.waitKey()


        evScoreAlexNet, imHolderAlexNet, imHolderAlexNetEarly = runKerasAlexNetModel(trainX, testX, trainY, testY, mode)

        print("AlexNet    |", evScoreAlexNet[1])
        cv2.imshow("AlexNet", imHolderAlexNet)
        cv2.imshow("AlexNetEarly", imHolderAlexNetEarly)
        del (evScoreAlexNet, imHolderAlexNet)
        gc.collect()

        evScoreOwn, imHolderOwn, imHolderOwnEarly = runOwnModel(trainX, testX, trainY, testY, mode)

        print("Own     |", evScoreOwn[1])
        cv2.imshow("Own", imHolderOwn)
        cv2.imshow("OwnEarly", imHolderOwnEarly)
        del (evScoreOwn, imHolderOwn)
        gc.collect()
        

        evScoreResNet, imHolderResnet,imHolderResnetEarly = runResnetKerasModel(trainX, testX, trainY, testY, mode)

        print("Resnet  |", evScoreResNet[1])
        cv2.imshow("Resnet", imHolderResnet)
        cv2.imshow("ResnetEarly", imHolderResnetEarly)
        del(evScoreResNet,imHolderResnet)
        gc.collect()




def createKerasResnetModel():
    base_model = ResNet101V2(weights='imagenet',
                             include_top=False,
                             input_shape=(width, height, 3)
                             )

    base_model.trainable = False

    model = Sequential([
        base_model,

        Conv2D(164, kernel_size=(7, 7)),

        Flatten(),
        Dense(164, activation="softmax"),

    ])

    return model


def createKerasVGG19Model():
    base_model = VGG19(weights='imagenet',
                       include_top=False,
                       input_shape=(width, height, 3),

                       )
    base_model.trainable = False

    model = Sequential([
        base_model,
        Conv2D(164, (3, 3), name="firstBreak", activation='ReLU'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(164, (2, 2), name="secondBreak", activation='ReLU'),
        Flatten(),
        Dense(164, activation="softmax"),

    ])
    return model


def createOwnModel():
    model = Sequential([

        Conv2D(96, (11, 11), input_shape=(width, height, 3), ),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # 2. layer
        Conv2D(128, (5, 5)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # 3. layer
        ZeroPadding2D((1, 1)),
        Conv2D(64, (4, 4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(3, 3)),

        Conv2D(64, (4, 4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3)),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(164, (4, 4)),
        BatchNormalization(),
        Activation('relu'),

        Flatten(),

        Dense(164, activation="softmax"),

    ])

    return model


def createKerasAlexNetModel():
    model = Sequential([
        # 1. layer
        Conv2D(96, (11, 11), input_shape=(width, height, 3),
               padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # 2. layer
        Conv2D(128, (5, 5), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # 3. layer
        ZeroPadding2D((1, 1)),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # 4. layer
        ZeroPadding2D((1, 1)),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        # 5. layer
        ZeroPadding2D((1, 1)),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),  # 21

        # 6. layer
        Flatten(),
        Dense(3072),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        # 7. layer
        Dense(4096),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        # 8. layer
        Dense(164),
        BatchNormalization(),  # 32
        Activation('softmax'),

    ])

    return model


def runKerasAlexNetModel(trainX, testX, trainY, testY, mode):
    modelKeras = load_model("pretrained/AlexNet.h5")
    tempArrHolder = []

    tempOut = [modelKeras.layers[32].output]

    functors = [K.function([modelKeras.input], [out]) for out in tempOut]

    for x in testX:
        x = x[np.newaxis, ...]
        tempArrHolder.append(functors[0](x))

    tempArrHolder = np.reshape(tempArrHolder, newshape=(-1, 164))

    tempImg = makePredictionWithImg(tempArrHolder, mode, testX, testY)

    evScore = modelKeras.evaluate(testX, testY)

    predictedValue = modelKeras.predict(testX, mode)

    imgHolder = makePredictionWithImg(predictedValue, mode, testX, testY)

    return evScore, imgHolder, tempImg


def runKerasVGGModel(trainX, testX, trainY, testY, mode):
    modelKeras = load_model("pretrained/VGG.h5")

    modelKeras.summary()
    tempArrHolder = []

    tempOut = [modelKeras.layers[3].output]

    functors = [K.function([modelKeras.input], [out]) for out in tempOut]

    for x in testX:
        x = x[np.newaxis, ...]
        tempArrHolder.append(functors[0](x))

    tempArrHolder = np.reshape(tempArrHolder, newshape=(-1, 164))

    tempImg = makePredictionWithImg(tempArrHolder, mode, testX, testY)

    evScore = modelKeras.evaluate(testX, testY)

    predictedValue = modelKeras.predict(testX, mode)

    imgHolder = makePredictionWithImg(predictedValue, mode, testX, testY)

    return evScore, imgHolder, tempImg


def runOwnModel(trainX, testX, trainY, testY, mode):
    tempArrHolder = []

    modelKeras = load_model("pretrained/Own.h5")

    tempOut = [modelKeras.layers[22].output]

    functors = [K.function([modelKeras.input], [out]) for out in tempOut]

    for x in testX:
        x = x[np.newaxis, ...]
        tempArrHolder.append(functors[0](x))

    tempArrHolder = np.reshape(tempArrHolder, newshape=(-1, 164))

    tempImg = makePredictionWithImg(tempArrHolder, mode, testX, testY)

    evScore = modelKeras.evaluate(testX, testY)

    predictedValue = modelKeras.predict(testX, mode)

    imgHolder = makePredictionWithImg(predictedValue, mode, testX, testY)

    return evScore, imgHolder, tempImg


def runResnetKerasModel(trainX, testX, trainY, testY, mode):
    modelKeras = load_model("pretrained/ResNet.h5")

    tempArrHolder = []

    tempOut = [modelKeras.layers[2].output]

    functors = [K.function([modelKeras.input], [out]) for out in tempOut]

    for x in testX:
        x = x[np.newaxis, ...]
        tempArrHolder.append(functors[0](x))

    tempArrHolder = np.reshape(tempArrHolder, newshape=(-1, 164))

    tempImg = makePredictionWithImg(tempArrHolder, mode, testX, testY)

    evScore = modelKeras.evaluate(testX, testY)

    predictedValue = modelKeras.predict(testX, mode)

    imgHolder = makePredictionWithImg(predictedValue, mode, testX, testY)

    return evScore, imgHolder, tempImg


def makePredictionWithImg(predictedValue, mode, testX, testY):
    simPic = [distance.cosine(predictedValue[whichPhoto], img) for img in predictedValue]

    print("Index of the photo : ", whichPhoto)

    kipkop = sorted(range(len(simPic)), key=lambda k: simPic[k])
    if (mode == 2):
        bipbop = createBipBop(kipkop, testY)
    else:
        bipbop = sorted(range(len(simPic)), key=lambda k: simPic[k])[1:6]

    # add if bipbop[n] == 0 print noFound.jpg

    firstCon = cv2.hconcat([testX[whichPhoto], testX[bipbop[0]]])
    secCon = cv2.hconcat([testX[bipbop[1]], testX[bipbop[2]]])
    thrdCon = cv2.hconcat([testX[bipbop[3]], testX[bipbop[4]]])

    sumCon = cv2.vconcat([firstCon, secCon])
    sumCon2 = cv2.vconcat([sumCon, thrdCon])

    return sumCon2


def createBipBop(fullbop, testY):  # name veriables like normal human beign at some point

    tempCounter = 0

    bipbop = [0, 0, 0, 0, 0]

    for x in range(1, len(fullbop)):

        tempClass = belongedClass(fullbop[x], testY, bipbop)
        for y in range(len(bipbop)):
            if (bipbop[tempClass] == 0):
                bipbop[tempClass] = fullbop[x]

                tempCounter += 1
        if (tempCounter == 4):
            break

    return bipbop


def belongedClass(temp, testY, bipbop):
    valueToReturn = -1

    temp = testY.iloc[temp]

    # overLayer = [10,15,16,17,52,59,71,89,90,97,102,131,132]
    # fullBody = [1,2,3,21,22,58,64,65,94,95,98,151,160,161,]

    accesories = [24, 25, 26, 27, 28, 29, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 73, 74, 75, 84, 103, 112, 113,
                  114, 115, 116, 117, 118, 122, 123, 124, 128, 137, 138, 139, 142, 143, 144, 154, 155, 157, 158, 159]
    top = [0, 9, 11, 12, 13, 14, 48, 60, 62, 63, 67, 72, 85, 86, 87, 88, 99, 120, 125, 130, 133, 147, 148, 149, 150,
           152,
           163]
    sumOfFullLayer = [0, 1, 2, 3, 10, 15, 16, 17, 21, 22, 23, 52, 58, 59, 64, 65, 71, 89, 90, 94, 95, 97, 98, 102, 126,
                      127, 131, 132, 145, 146, 151, 160, 161]
    down = [0, 4, 5, 6, 7, 8, 18, 19, 20, 53, 54, 55, 56, 57, 61, 66, 68, 69, 70, 91, 92, 93, 96, 100, 101, 119, 121,
            129, 134, 135, 153, 162]
    shoes = [30, 31, 32, 33, 34, 35, 36, 37, 49, 50, 76, 77, 78, 79, 80, 81, 82, 83, 104, 105, 106, 107, 108, 109, 110,
             111, 136, 140, 141, 156]

    for x in range(len(accesories)):
        if (accesories[x] == temp):
            valueToReturn = 0
    for x in range(len(top)):
        if (top[x] == temp):
            valueToReturn = 1
    for x in range(len(sumOfFullLayer)):
        if (sumOfFullLayer[x] == temp):
            valueToReturn = 2
    for x in range(len(down)):
        if (down[x] == temp):
            valueToReturn = 3
    for x in range(len(shoes)):
        if (shoes[x] == temp):
            valueToReturn = 4

    if (valueToReturn == -1):
        print("You forgot to implement : ", temp)

    return valueToReturn

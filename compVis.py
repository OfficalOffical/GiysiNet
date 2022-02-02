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
from keras.layers import GlobalMaxPooling2D
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import pandas as pd

def getImageFromDest(datasetPath):
    tempImage = []


    w = 128
    h = 128
    resizeRate = (w, h)

    for i in os.listdir(datasetPath):
        for file in  glob.glob(datasetPath + "/" + i + "/*.jpg"):
            tempImage.append(cv2.resize(cv2.imread(file),resizeRate))

    """ 
    tempSizeHolder = 0
    for i in tempImage:
        image.append([cv2.resize(temp, resizeRate) for temp in i])"""

    image = np.array(tempImage)
    return image


def showImage(img, i):
    cv2.imshow("a", img[i])
    cv2.waitKey(0)


def kerasModel(img):
    img_width, img_height, _ = 128, 128, 3  # load_image(df.iloc[0].image).shape

    # Pre-Trained Model
    base_model = ResNet50(weights='imagenet',
                          include_top=False,
                          input_shape=(img_width, img_height, 3))
    base_model.trainable = False

    # Add Layer Embedding
    model = keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])

    model.summary()


def embedding(model, img):
    a = np.array(img, dtype=object)
    a = a.flatten()
    x = image.img_to_array(a)


def getImageFromJSON(jsonPath):
    jsonPath = jsonPath + "/train_no_dup.json"
    temp = pd.read_json(jsonPath) # <<< burayı düzelt
    temp = temp.head(100)
    temp = temp.drop(labels=["name","image","set_url","views", "likes", "date", "desc"], axis=1)
    temp = temp.sort_values(by=['set_id'])
    return temp

"""def downloadImg():
    urllib.request.urlretrieve("http://ak1.polyvoreimg.com/cgi/img-set/cid/214181831/id/El8a99fQ5hG4HrPFO4xqOQ/size/y.jpg", "00000001.jpg")"""


def getLabeled(img):



    """datasetPath = "C:/Users/Sefa/Desktop/db"

    tempImage = []

    w = 128
    h = 128
    resizeRate = (w, h)

    for i in os.listdir(datasetPath):
        if else <<
        for file in glob.glob(datasetPath + "/" + i + "/*.jpg"):
            tempImage.append(cv2.resize(cv2.imread(file), resizeRate)"""

    for x in range(len(img)):
        for items in img["items"][x]:
            return 0
        #print(img["set_id"][x])
        print(img["items"][x][0]['categoryid'])

    return 0

#http://www.polyvore.com/being_vans_shoe_model_with/set?id=120161271
#http://ak1.polyvoreimg.com/cgi/img-set/cid/214181831/id/El8a99fQ5hG4HrPFO4xqOQ/size/y.jpg
















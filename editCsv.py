import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd


def notAvailable(newList, j):
    newJ = ""
    for k in range(len(j)):
        if k > 0:
            if k == 1:
                newJ += str(j[k])
            else:
                newJ += " " + str(j[k])

    for x in range(len(newList)):
        if newList[x] == newJ:
            return False
    return True


def getImageFromDest(datasetPath, csv, width, height):
    tempImage = []
    tempSetId = []
    resizeRate = (width, height)

    for i in range(len(csv)):
        file = datasetPath + "/" + str(csv['set_id'][i]) + "/" + str(csv['index'][i]) + ".jpg"
        tempSetId.append(csv['set_id'][i])
        tempImage.append(cv2.resize(cv2.imread(file), resizeRate))

    image = np.array(tempImage)
    setId = np.array(tempSetId)

    return image


def getCsv(csvPath, w, h, nRowSetter):
    print("Started reading images")
    datasetPath = "E:/db"
    temp = pd.read_csv(csvPath, nrows=nRowSetter)  # <<< burayı düzelt

    temp = temp.drop(labels=["id", "color", "sub_category", "name", "all"], axis=1)
    temp = temp.sort_values(by=['set_id', 'index'], ignore_index=True)
    imageArray = getImageFromDest(datasetPath, temp, w, h)
    print("Finished reading images")
    return temp, imageArray


def getAndCleanCsv(csvPath, w, h, nRowSetter):
    temp, imgArr = getCsv(csvPath, w, h, nRowSetter)

    temp['tempCategoryId'] = temp['categoryx_id']

    print("Normalising Dataset Started")
    with open('C:/Users/Sefa/Desktop/oldCategory.txt') as f:
        lines = f.readlines()

    for x in range(len(temp["categoryx_id"])):
        for j in range(len(lines)):
            tempStr = lines[j].split()
            if int(temp["categoryx_id"][x]) == int(tempStr[0]):
                temp["categoryx_id"][x] = j


    print("Normalising Dataset Finished")

    return temp, imgArr



"""
a = [141,99,95,107,63,106,84,150,127,100,100,103,149,89,96,49,119,71,94,144,142,67,
     140,23,48,136,151,111,126,90,114,92,93,133,70,138,86,120,158,65,87,125,112,22,
     155,108,82,75,41,159,156,145,64,62,137,131,143,153,55,109,135,115,163,128,123,
     83,157,72,54,85]
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd





def notAvailable(newList, j):
    newJ = ""
    for k in range(len(j)):
        if (k > 0):
            if (k == 1):
                newJ += str(j[k])
            else:
                newJ += " " + str(j[k])


    for x in range(len(newList)):
        if (newList[x] == newJ):
            return False
    return True

def getImageFromDest(datasetPath,csv,width,height):

    tempImage = []
    tempSetId = []
    resizeRate = (width, height)

    for i in range(len(csv)):
            file = datasetPath+"/"+str(csv['set_id'][i]) + "/" + str(csv['index'][i])+".jpg"
            tempSetId.append(csv['set_id'][i])
            tempImage.append(cv2.resize(cv2.imread(file), resizeRate))


    image = np.array(tempImage)
    setId = np.array(tempSetId)




    return image


def getCsv(csvPath,w,h,nRowSetter):
    print("Started reading images")
    datasetPath = "E:/db"
    temp = pd.read_csv(csvPath,nrows=nRowSetter)  # <<< burayı düzelt


    temp = temp.drop(labels=["id","color", "sub_category", "name", "all"], axis=1)
    temp = temp.sort_values(by=['set_id', 'index'],ignore_index=True)
    imageArray = getImageFromDest(datasetPath,temp,w,h)
    print("Finished reading images")
    return temp,imageArray


def getAndCleanCsv(csvPath,w,h,nRowSetter):

    temp,imgArr = getCsv(csvPath,w,h,nRowSetter)

    tempHolder =0

    print("Normalising Dataset Started")
    with open('C:/Users/Sefa/Desktop/oldCategory.txt') as f:
        lines = f.readlines()

    for x in range(len(temp["categoryx_id"])):
        for j in range(len(lines)):
            tempStr = lines[j].split()
            if (int(temp["categoryx_id"][x]) == int(tempStr[0])):
                temp["categoryx_id"][x] = j


    plt.figure(figsize=(7, 25))
    temp.categoryx_id.value_counts().sort_values().plot(kind='barh')
    plt.show()





    print("Normalising Dataset Finished")



    return temp,imgArr


























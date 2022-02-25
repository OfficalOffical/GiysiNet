import cv2
import matplotlib.pyplot as plt
import numpy as np
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

    resizeRate = (width, height)

    for i in range(len(csv)):
            file = datasetPath+"/"+str(csv['set_id'][i]) + "/" + str(csv['index'][i])+".jpg"
            tempImage.append(cv2.resize(cv2.imread(file), resizeRate))


    image = np.array(tempImage)
    return image


def getCsv(csvPath,w,h,nRowSetter):
    print("Started reading images")
    datasetPath = "E:/db"
    temp = pd.read_csv(csvPath,nrows=nRowSetter)  # <<< burayı düzelt


    temp = temp.drop(labels=["id","color", "sub_category", "name", "all"], axis=1)
    temp = temp.sort_values(by=['set_id', 'index'])
    imageArray = getImageFromDest(datasetPath,temp,w,h)
    print("Finished reading images")
    return temp,imageArray


def getAndCleanCsv(csvPath,w,h,nRowSetter):

    temp,imgArr = getCsv(csvPath,w,h,nRowSetter)




    print("Normalising Dataset Started")
    with open('C:/Users/Sefa/Desktop/oldCategory.txt') as f:
        lines = f.readlines()

    for x in range(len(temp["categoryx_id"])):
        if(labelNormaliser(temp["categoryx_id"][x]) == False):
            temp.drop(x,axis = 0, inplace = True)
            imgArr = np.delete(imgArr,x,axis=0)


        else:
            for j in range(len(lines)):
                tempStr = lines[j].split()
                if (int(temp["categoryx_id"][x]) == int(tempStr[0])):
                    temp["categoryx_id"][x] = j

    print("Normalising Dataset Finished")

    plt.figure(figsize=(7,20))
    temp.categoryx_id.value_counts().sort_values().plot(kind='barh')
    plt.show()

    return temp,imgArr

def labelNormaliser(datasetInput):
    datasetInput = int(datasetInput)



    switcher = {
        64 : True,
        37 : True,
        62 : True,
        11 : True,
        57 : True,
        261 : True,
        43 : True,
        46 : True,
        38 : True,
        65 : True,
        25 : True,
        49 : True,
        19 : True,
        4 : True,
        237 : True,
        24 : True,
        318 : True,
        29 : True,
        21 : True,
        55 : True,
        9 : True,
        17 : True,
        36 : True,
        259 : True,
        61 : True,
        28 : True,
        2 : True,
        5 : True,
        8 : True,
        47 : True,
        41 : True,
        27 : True,
        236 : True,
        42 : True,
        240 : True,
        4495 : True,
        18 : True,
        241 : True,
        60 : True,
        40 : True,
        4428 : True,
        52 : True,
        3 : True,
        265 : True,
        6 : True,


    }
    return switcher.get(datasetInput,False)















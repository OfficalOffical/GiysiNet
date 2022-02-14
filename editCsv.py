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



def getAndCleanCsv(csvPath):
    temp = pd.read_csv(csvPath, nrows=200000)

    with open('C:/Users/Sefa/Desktop/oldCategory.txt') as f:
        lines = f.readlines()

    counter = 0
    newList = []
    tempStr = ""

    for x in range(len(temp["categoryx_id"])):
        for j in lines:
            j = j.split()
            if (int(temp["categoryx_id"][x]) == int(j[0])):
                for k in range(len(j)):
                    if (k>0 and notAvailable(newList,j)):
                        if(k == 1):
                            tempStr += str(j[k])
                        else:
                            tempStr += " " + str(j[k])

                if (tempStr != ""):
                    newList.append(tempStr)

                print(len(newList))

                tempStr=""

                """  
                temp=None"""
    return newList














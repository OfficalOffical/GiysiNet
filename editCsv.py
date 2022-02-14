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



def getAndCleanCsv(temp):


    with open('C:/Users/Sefa/Desktop/oldCategory.txt') as f:
        lines = f.readlines()




    for x in range(len(temp["categoryx_id"])):
        for j in range(len(lines)):
            tempStr = lines[j].split()
            if (int(temp["categoryx_id"][x]) == int(tempStr[0])):
                temp["categoryx_id"][x] = j

    return temp















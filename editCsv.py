import pandas as pd

def getAndCleanCsv(csvPath):
    temp = pd.read_csv(csvPath, nrows=1000)

    with open('C:/Users/Sefa/Desktop/oldCategory.txt') as f:
        lines = f.readlines()

    counter = 0

    for x in range(len(temp["categoryx_id"])):
        for j in lines:
            j = j.split()
            if (int(temp["categoryx_id"][x]) == int(j[0])):
                for k in range(len(j)):
                    if(k != 0):
                        print(j[k])
                print()



    print(temp)





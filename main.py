from compVis import *


datasetPath = "E:/db"
csvPath = "C:/Users/Sefa/Desktop/outfit_data_cleaned.csv"



tempHolder = 0
mode = 2

"""
while ( tempHolder == 0):
    mode = int(input( "Aynı kategoride benzerlik tavsiyesi için 1 farklı kategorilerde ilişkisel tavsiye için lütfen 2 girdisi veriniz : "))
    if(mode != 1 or mode != 2):
        tempHolder = 1"""


a = getImageFromCSV(csvPath,mode )













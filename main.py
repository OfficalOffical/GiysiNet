from compVis import *


datasetPath = "db" #polyvore dataset
csvPath = "outfit_Data_cleaned.csv"



tempHolder = 0



while ( tempHolder == 0):
    mode = int(input( "Aynı kategoride benzerlik tavsiyesi için 1,\nfarklı kategorilerde "
                      " ilişkisel tavsiye için 2,\ntek model ile hızlı ilişkisel sonuç için"
                      " lütfen 3 girdisi veriniz : "))
    if(mode != 1 or mode != 2 or mode != 3):
        tempHolder = 1


a = getImageFromCSV(csvPath,mode )













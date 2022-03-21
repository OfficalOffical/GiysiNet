import cv2
import matplotlib.pyplot as plt
from keras.applications.resnet_v2 import ResNet50V2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.optimizers import adam_v2
import pandas as pd
import editCsv
from scipy.spatial import distance



width = 128
height = 128
nRowSetter = 10000










def getImageFromCSV(csvPath):
    csv,imarr = editCsv.getAndCleanCsv(csvPath,width,height,nRowSetter)
    modelKeras = createKerasModel()





    trainX, testX, trainY, testY = train_test_split(imarr,csv["categoryx_id"],test_size=0.3,random_state=42)
    # u can devide it to 255 to make it 0-1 in further i guess dk







    print(modelKeras.summary())



    print(trainX.shape,testX.shape,trainY.shape,testY.shape)

    #0.001 def
    opt = adam_v2.Adam(learning_rate=0.0001)

    modelKeras.compile(loss='sparse_categorical_crossentropy',optimizer= opt ,metrics=['sparse_categorical_accuracy'])


    modelKeras.fit(trainX,trainY,epochs=15)

    modelKeras.evaluate(testX, testY)



    pp = modelKeras.predict(testX)




    #can be lower the features with PCA i guess


    whichPhoto = 25

    simPic = [ distance.cosine(pp[whichPhoto],img )for img in pp] # 25. fotoyla cosine sim yapıyor

    bipbop = sorted(range(len(simPic)),key=lambda k:simPic[k])[1:6] #ilk 6 datayı sortlayıp yazdırıyor ilki kendisi olduğu için pass







    firstCon = cv2.hconcat([testX[whichPhoto], testX[bipbop[0]]])
    secCon = cv2.hconcat([testX[bipbop[1]], testX[bipbop[2]]])
    thrdCon = cv2.hconcat([testX[bipbop[3]], testX[bipbop[4]]])

    sumCon = cv2.vconcat([firstCon, secCon])
    sumCon2 = cv2.vconcat([sumCon, thrdCon])

    cv2.imshow("A",sumCon2)
    cv2.waitKey()


    plt.figure(figsize=(16,4))
    plt.plot(pp[0])
    plt.show()




    """  
    mnist %99 acc
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    print(modelKeras.summary())


    print(train_X.shape,train_y.shape,test_X.shape,test_y.shape)


    modelKeras.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['sparse_categorical_accuracy'])

    modelKeras.fit(train_X,train_y,epochs=15)
    modelKeras.evaluate(test_X,test_y)"""


    return csv

def createKerasModel():
    # Input Shape
    base_model = ResNet50V2(weights='imagenet',
                            include_top=False,
                            input_shape=(width, height, 3))

    base_model.trainable = True

    model = Sequential([
        base_model,

        MaxPooling2D(pool_size=(2, 2)),


        Flatten(),  # Is it mandatory ? end to end conv2d ?

        Dense(164, activation="softmax"),
        # 1000 -> loss 3.85 \ acc 0.09
    ])



    """ model = Sequential([
    #128x128 acc %22
        keras.Input(shape=(width,height,3)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(2048, (8, 8), activation="relu"),

        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(1024, (6, 6), activation='relu'),

        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(512, (4, 4), activation='relu'),

        Conv2D(256, (3, 3), activation='relu'),

        Flatten(), # Is it mandatory ? end to end conv2d ?

        Dense(164, activation="softmax"),
        #1000 -> loss 3.85 \ acc 0.09
    ])"""






    return model




















import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pickle
from keras import Model

# Settings
path = "Data"
test_size = 0.2
val_size = 0.2

# code

images = []
classNo = []
myList = os.listdir(path=path)
noOfClasses = len(myList)

for x in range(0, noOfClasses):
    myPicList = os.listdir(path+"/"+str(x))

    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/" + y)
        curImg = cv2.resize(curImg, (32,32))
        images.append(curImg)
        classNo.append(x)
    
    print(x)

print(len(classNo) == len(images))

images = np.array(images)
classNo = np.array(classNo)

print(images.shape)
print(classNo.shape)

## Split data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=test_size)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size)

print(X_train.shape)

numOfSamples = []
for x in range(0,noOfClasses):
    numOfSamples.append(len(np.where(y_train == x)[0]))


plt.figure(figsize=(10,5))
plt.bar(range(0, noOfClasses), numOfSamples)
plt.title("Number of images for each class")
plt.xlabel("class id")
plt.ylabel("Number of images")

plt.show()

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img


# img = preProcessing(X_train[30])
# img = cv2.resize(img, (300,300))
# cv2.imshow("preprocessed", img)
# cv2.waitKey(0)

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_val = np.array(list(map(preProcessing, X_val)))

X_train = X_train.reshape((-1, 32,32, 1))
X_test = X_test.reshape((-1, 32,32, 1))
X_val = X_val.reshape((-1, 32,32, 1))


# Augmentation
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

dataGen.fit(X_train)

y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_val = to_categorical(y_val, noOfClasses)

def myModel():
    noOfFilters= 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)

    noOfNOde = 500
    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(32,32,1), activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation="relu")))
    model.add(MaxPooling2D(pool_size=sizeOfPool))

    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation="relu")))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNOde, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=['accuracy'])

    return model

model = myModel()
print(model.summary())

history = model.fit(
    dataGen.flow(X_train, y_train, batch_size=50),
    validation_data=(X_val, y_val),
    epochs=10,
    steps_per_epoch=2000,
    shuffle=1
)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.legend(['training', 'validation'])
plt.title("Loss")
plt.xlabel("epoch")

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.legend(['training', 'validation'])
plt.title("Accuracy")
plt.xlabel("epoch")

plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print("Test Score = ", score[0])
print("Test Accuracy = ", score[1])


# Saving
pickle_out = open("model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
import pickle
import os.path

import tkinter.messagebox
from tkinter import *
from tkinter import simpledialog, filedialog

import PIL
import PIL.Image, PIL.ImageDraw
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

proj_name = "ImageDataset"
class1 = "Triangle"
class2 = "Square"
class3 = "Circle"
class4 = "Arrow"
class5 = "Star"
testName = "TestData"


img_list = np.array([])
class_list = np.array([])
test_list = np.array([])
class1_counter = 50
class2_counter = 50
class3_counter = 50
class4_counter = 50
class5_counter = 50

#reading data in and reshaping data into arrays to be added to dataset array
#triangle values
for x in range(1, class1_counter):
    img = cv.imread(f"{proj_name}/{class1}/{x}.png")[:, :, 0]
    img = img.reshape(2500)
    img_list = np.append(img_list, [img])
    class_list = np.append(class_list, 1)

#square
for x in range(1, class2_counter):
    img = cv.imread(f"{proj_name}/{class2}/{x}.png")[:, :, 0]
    img = img.reshape(2500)
    img_list = np.append(img_list, [img])
    class_list = np.append(class_list, 2)
#circle
for x in range(1, class3_counter):
    img = cv.imread(f"{proj_name}/{class3}/{x}.png")[:, :, 0]
    img = img.reshape(2500)
    img_list = np.append(img_list, [img])
    class_list = np.append(class_list, 3)
#arrow
for x in range(1, class4_counter):
    img = cv.imread(f"{proj_name}/{class4}/{x}.png")[:, :, 0]
    img = img.reshape(2500)
    img_list = np.append(img_list, [img])
    class_list = np.append(class_list, 4)
#star
for x in range(1, class5_counter):
    img = cv.imread(f"{proj_name}/{class5}/{x}.png")[:, :, 0]
    img = img.reshape(2500)
    img_list = np.append(img_list, [img])
    class_list = np.append(class_list, 5)


#make list of images become two dimensional where each row becomes an image from a class
img_list = img_list.reshape(class1_counter - 1 + class2_counter - 1 + class3_counter-1 + class4_counter - 1 + class5_counter - 1, 2500)
#K and P values we can change in order to see their effects
neighbours = 1
pValue = 3

model = KNeighborsClassifier(n_neighbors=neighbours,weights='distance',algorithm='auto',leaf_size=30,p=pValue,metric='minkowski',metric_params=None,n_jobs=None)

accuracies = np.array([])
#splitting the data 80% training and 20% testing over 100 different splits of the dataset
for t in range(101):
    X_train, X_test, y_train, y_test = train_test_split(img_list, class_list, test_size=0.2)

    model.fit(X_train,y_train)
    predictionList = model.predict(X_test)

    accuracy = accuracy_score(y_test,predictionList)
    accuracies = np.append(accuracies,accuracy)

#getting min and max accuracies
min_value = np.min(accuracies)
max_value = np.max(accuracies)
print(min_value)
print(max_value)

#plotting boxplot graph to see variance of model
plt.boxplot(accuracies)
plt.xlabel(f"kNN Model with {neighbours} Nearest Neighbours and a P Value of {pValue}")
plt.ylabel("Accuracy")
plt.title("Accuracy and Variance over 100 Data Splits")
plt.show()


kValues = [i for i in range(1,101)]
#creating new split of data to be used for seeing k value effect on accuracy
X_train, X_test, y_train, y_test = train_test_split(img_list, class_list, test_size=0.2)
accuraciesk = np.array([])


#looping through range of k values to see how accuracy varies
for k in kValues:
    model1 = KNeighborsClassifier(n_neighbors=k,weights='distance',algorithm='auto',leaf_size=30,p=1,metric='minkowski',metric_params=None,n_jobs=None)

    model1.fit(X_train,y_train)
    predictionList = model1.predict(X_test)

    accuracy = accuracy_score(y_test,predictionList)
    accuraciesk = np.append(accuraciesk,accuracy)

#plotting how accuracy changes with k values
plt.plot(kValues,accuraciesk)
plt.ylim(0,1)
plt.xlabel("K Values")
plt.ylabel("Accuracy")
plt.title("Graph Showing Accuracy of Model for Different K Values")
plt.show()
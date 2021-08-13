
import numpy as np
import random
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import FL_utils


def simpleSGDTest():

    X_train, X_test, y_train, y_test = FL_utils.prepareTrainTest('trainingSet/trainingSet')

    return FL_utils.simpleSGD(X_train, y_train, X_test, y_test)



def cross_val_SGD(splitter = 5):

    image_list, label_list = FL_utils.prepareAllData('trainingSet/trainingSet')
    size = len(label_list)
    total_accuracy = 0

    for i in range(splitter):
        split = int(size / splitter)
        spliti = int(i * (size / splitter))
        X_train1 = image_list[0:spliti]  # 0:61
        y_train1 = label_list[0:spliti]  # 0:61
        print("train 1: " + str(0) + " à " + str(spliti))

        X_test = image_list[spliti:spliti + split]  # 61:122
        y_test = label_list[spliti:spliti + split]  # 61:122
        print("test: " + str(spliti) + " à " + str(spliti + split))
        X_train2 = image_list[spliti + split:]
        y_train2 = label_list[spliti + split:]
        print("train 2: " + str(spliti + split) + " à " + "la fin")

        X_train = np.append(X_train1, X_train2, axis=0)
        y_train = np.append(y_train1, y_train2, axis=0)

        clients = FL_utils.create_clients(X_train, y_train, num_clients=10)

        local_accuracy = FL_utils.simpleSGD(X_train, y_train, X_test, y_test)
        total_accuracy += local_accuracy
    return total_accuracy / splitter
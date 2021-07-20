
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import random
import math
from matplotlib import pyplot
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

import torch
import math
import torch.nn.functional as F
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from pathlib import Path
import requests
import pickle
import gzip

import math

pd.options.display.float_format = "{:,.4f}".format

((x_train, y_train), (x_test, y_test)) = mnist.load_data()


def showDigits():
    # Print some digits
    fig, axes = pyplot.subplots(8,8,figsize=(8,8))
    for i in range(8):
        for j in range(8):
            num_index = np.random.randint(len(x_train))
            axes[i,j].imshow(x_train[num_index].reshape((28, 28)), cmap="gray")
            axes[i,j].axis("off")
    pyplot.show()

def countTags():
    # Let's check how many of each tag are.
    y_train_total = 0
    y_test_total = 0
    total = 0
    for i in range(10):
        print(i, ">> train:", sum(y_train == i), "test:", sum(y_test == i), ", total:", sum(y_train == i) + sum(y_test == i))
        y_train_total = y_train_total + sum(y_train == i)
        y_test_total = y_test_total + sum(y_test == i)
        total = total + sum(y_train == i) + sum(y_test == i)

    print("y_train_total=", y_train_total)
    print("y_test_total=", y_test_total)
    print("total=", total)


"""  
@input  y_data: numpy array of the labels
        seed: parameter to control randomness
        amount: number of indexes for each label
@output label_dict: dictionary of dataframes of indexes of each label
"""
def split_and_shuffle_labels(y_data, seed, amount):
    # Shuffle the indexes of the labels in order to distribute the data as IID
    y_data = pd.DataFrame(y_data, columns=["labels"])
    y_data["i"] = np.arange(len(y_data))
    label_dict = dict()
    for i in range(10):
        var_name = "label" + str(i)
        label_info = y_data[y_data["labels"] == i]
        np.random.seed(seed)
        label_info = np.random.permutation(label_info)
        label_info = label_info[0:amount]
        label_info = pd.DataFrame(label_info, columns=["labels", "i"])
        label_dict.update({var_name: label_info})
    return label_dict

"""  
@input  label_dict: dictionary of dataframes of indexes of each label
        number_of_samples: Number of samples created
        amount: number of indexes for each label
@output sample_dict: dictionary of indexes of labels, identically distributed
"""
def get_iid_subsamples_indices(label_dict, number_of_samples, amount):
    # divides the indexes in each node with an equal number of each label
    sample_dict = dict()
    batch_size = int(math.floor(amount / number_of_samples))
    for i in range(number_of_samples):
        sample_name = "sample" + str(i)
        dumb = pd.DataFrame()
        for j in range(10):
            label_name = str("label") + str(j)
            a = label_dict[label_name][i * batch_size:(i + 1) * batch_size]
            dumb = pd.concat([dumb, a], axis=0)
        dumb.reset_index(drop=True, inplace=True)
        sample_dict.update({sample_name: dumb})
    return sample_dict

"""  
@input  sample_dict: dictionary of indexes of labels, identically distributed
        x_data: data of x
        y_data: data of y
        x_name: name of x
        y_name: name of y
@output x_data_dict: dictionary of data x for each node with keys "x_trainX"
        y_data_dict dictionary of data y for each node with keys "y_trainX"
"""
def create_iid_subsamples(sample_dict, x_data, y_data, x_name, y_name):
    # This function distributes x and y data to nodes in dictionary
    x_data_dict = dict()
    y_data_dict = dict()

    for i in range(len(sample_dict)):  ### len(sample_dict)= number of samples
        xname = x_name + str(i)
        yname = y_name + str(i)
        sample_name = "sample" + str(i)

        indices = np.sort(np.array(sample_dict[sample_name]["i"]))

        x_info = x_data[indices, :]
        x_data_dict.update({xname: x_info})

        y_info = y_data[indices]
        y_data_dict.update({yname: y_info})

    return x_data_dict, y_data_dict

number_of_samples=100
learning_rate = 0.01
numEpoch = 10
batch_size = 32
momentum = 0.9

train_amount=5400
valid_amount=900
test_amount=900
print_amount=3

label_dict_train=split_and_shuffle_labels(y_data=y_train, seed=1, amount=train_amount)
# print(label_dict_train)

sample_dict_train=get_iid_subsamples_indices(label_dict=label_dict_train, number_of_samples=number_of_samples, amount=train_amount)
# print(sample_dict_train)

x_train_dict, y_train_dict = create_iid_subsamples(sample_dict=sample_dict_train, x_data=x_train, y_data=y_train, x_name="x_train", y_name="y_train")
print(x_train_dict['x_train1'])


# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


# define cnn model
def define_model():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(200, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(200, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        # define model
        model = define_model()
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        model.summary()
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories


# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    pyplot.show()


# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))
    # box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.show()


# run the test harness for evaluating a model
def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # evaluate model
    scores, histories = evaluate_model(trainX, trainY)
    # learning curves
    summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)


# entry point, run the test harness
#run_test_harness()



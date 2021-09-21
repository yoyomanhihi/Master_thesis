import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
import medical_preprocessing_2d as med_prep_2d
import medical_preprocessing_3d as med_prep_3d
import sys
import cv2
import os

np.set_printoptions(threshold=sys.maxsize)


def load(path):
    ''' Load the pickle data. The data already contains the 32x32 (32x32x32 in case of 3d) and a correct binary classification'''
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data


def prepareTrainTest_2d(path):
    ''' Open the data and prepare it for ML
        args:
            path: path to the file where the data is stored
        return:
            x_train: training images
            y_train: training labels
            x_test: test images
            y_test: test labels'''
    data = load(path)
    data = np.array(data)
    np.random.shuffle(data)
    size = len(data)
    train_size = int(0.8*size)
    inputs = []
    outputs = []
    for elem in data:
        # Set the coordonates of the image between 0 and 1
        elem[0][1024] = elem[0][1024]/480
        elem[0][1025] = elem[0][1025]/480
        # Add the inputs and outputs to the data
        inputs.append(elem[0])
        outputs.append(elem[1])
    x_train = inputs[:train_size]
    y_train = outputs[:train_size]
    x_test = inputs[train_size:]
    y_test = outputs[train_size:]
    y_train = np.asarray(y_train).astype('float32').reshape((-1, 1))
    y_test = np.asarray(y_test).astype('float32').reshape((-1, 1))
    return x_train, y_train, x_test, y_test


def prepareTrainTest_3d(path):
    ''' Open the data and prepare it for ML
        args:
            path: path to the file where the data is stored
        return:
            x_train: training images
            y_train: training labels
            x_test: test images
            y_test: test labels'''
    data = load(path)
    data = np.array(data)
    np.random.shuffle(data)
    size = len(data)
    train_size = int(0.8*size)
    inputs = []
    outputs = []
    for elem in data:
        # Set the coordonates of the image between 0 and 1
        elem[0][32768] = elem[0][32768]/600 #600 seems to be a good choice
        elem[0][32769] = elem[0][32769]/480
        elem[0][32770] = elem[0][32770]/480
        # Add the inputs and outputs to the data
        inputs.append(elem[0])
        outputs.append(elem[1])
    x_train = inputs[:train_size]
    y_train = outputs[:train_size]
    x_test = inputs[train_size:]
    y_test = outputs[train_size:]
    y_train = np.asarray(y_train).astype('float32').reshape((-1, 1))
    y_test = np.asarray(y_test).astype('float32').reshape((-1, 1))
    return x_train, y_train, x_test, y_test


class SimpleMLP:
    ''' Build and return a simple MLP model'''
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("sigmoid"))
        return model


def bcr(TP, FP, TN, FN):
    ''' Compute the bcr score of the model
        args:
            TP: true positives
            FP: false positives
            TN: true negatives
            FN: false negatives'''
    left = 0
    if TP+FN > 0:
        left = TP / (TP + FN)
    right = 0
    if FP + TN > 0:
        right = TN / (FP + TN)
    return (left + right) / 2


def test_model(x_test, y_test, model):
    """ Calculates the accuracy and the loss of the model
        args:
            x_test: test set data
            y_test: test set labels
            model: the model
        return:
            acc: accuracy of the model

    """
    test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))
    for (X_test, Y_test) in test_batched:
        print("shape: " + str(np.shape(X_test)))
        predictions = model.predict(X_test)
    print(predictions)
    print(y_test)
    for i in range(len(predictions)):
        if predictions[i] > 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for k in range(len(predictions)):
        if predictions[k] == 0:
            if y_test[k] == 0:
                TN += 1
            else:
                FN += 1
        if predictions[k] == 1:
            if y_test[k] == 1:
                TP += 1
            else:
                FP += 1
    acc = bcr(TP, FP, TN, FN)
    print("Finale accuracy = " + str(acc))
    return acc


def simpleSGD_2d(x_train, y_train, x_test, y_test, lr = 0.01, comms_round = 100):
    ''' Simple SGD algorithm for 32x32 images
        args:
            x_train: training images
            y_train: training labels
            x_test: test images
            y_test: test labels
            lr: learning rate
            comms_round: number of communication rounds
        returns:
            SGD_model: model fitted
    '''

    loss = 'binary_crossentropy'
    metrics = ['accuracy']
    optimizer = SGD(lr=lr,
                    decay=lr / comms_round,
                    momentum=0.9
                    )

    SGD_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(y_train)).batch(320)
    smlp_SGD = SimpleMLP()
    SGD_model = smlp_SGD.build(1026, 1)

    SGD_model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    # fit the SGD training data to model
    _ = SGD_model.fit(SGD_dataset, epochs=100, verbose=2)

    #test the SGD global model and print out metrics

    SGD_acc = test_model(x_test, y_test, SGD_model)

    return SGD_model


def simpleSGD_3d(X_train, y_train, X_test, y_test, lr = 0.01, comms_round = 100):
    ''' Simple SGD algorithm for 32x32x32 images
        args:
            x_train: training images
            y_train: training labels
            x_test: test images
            y_test: test labels
            lr: learning rate
            comms_round: number of communication rounds
        returns:
            SGD_model: model fitted
    '''

    loss = 'binary_crossentropy'
    metrics = ['accuracy']
    optimizer = SGD(lr=lr,
                    decay=lr / comms_round,
                    momentum=0.9
                    )

    SGD_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(320)
    smlp_SGD = SimpleMLP()
    SGD_model = smlp_SGD.build(32771, 1)

    SGD_model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    # fit the SGD training data to model
    _ = SGD_model.fit(SGD_dataset, epochs=100, verbose=2)

    SGD_acc = test_model(X_test, y_test, SGD_model)

    return SGD_model


def crop_2d(image, y, x): #vertical, horizontal
    ''' Return a 32x32 image with top left corner of coordonate(y, x)'''
    crop_img = image[y:y + 32, x:x + 32]
    return crop_img


def segmentation_2d(model, image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    predictions = np.zeros((512, 512))
    image = image/255
    for y in range(0, 480, 10):
        print("predicting line: " + str(y))
        for x in range(0, 480, 10):
            flatten_subimage = crop_2d(image, y, x).flatten()
            flatten_subimage = np.append(flatten_subimage, y/480)
            flatten_subimage = np.append(flatten_subimage, x/480)
            reshaped = np.reshape(flatten_subimage, (1,1026))
            pred = model.predict(reshaped)
            if pred[0] > 0.5:
                print("FOUND A TUMOR HERE GUYS")
                predictions[y:y + 32, x:x + 32] += 1
    print(predictions)
    return predictions


def crop_3d(image, z, y, x): #vertical, horizontal
    ''' Return a 32x32 image with top left corner of coordonate(y, x) '''
    crop_img = image[z: z + 32, y:y + 32, x:x + 32]
    return crop_img


def segmentation_3d(model, images_path):
    predictions = np.zeros((512, 512, 512))
    image_3d = []
    size = len(os.listdir(images_path))
    for i in range(size):
        image_path = images_path + "/image_" + str(i) +".png"
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image/255
        image_3d.append(image)
    image_3d = np.array(image_3d)
    depth = len(image_3d)
    for z in range(0, depth-32, 10):
        print("predicting depth: " + str(z))
        for y in range(0, 480, 10):
            for x in range(0, 480, 10):
                flatten_subimage = crop_3d(image_3d, z, y, x).flatten()
                flatten_subimage = np.append(flatten_subimage, z/600)
                flatten_subimage = np.append(flatten_subimage, y/480)
                flatten_subimage = np.append(flatten_subimage, x/480)
                reshaped = np.reshape(flatten_subimage, (1,32771))
                pred = model.predict(reshaped)
                if pred[0] > 0.5:
                    print("FOUND A TUMOR HERE GUYS")
                    predictions[z:z + 32, y:y + 32, x:x + 32] += 1
    print(predictions)
    return predictions






""" Federated learning code inspired by this tutorial:
https://towardsdatascience.com/federated-learning-a-step-by-step-implementation-in-tensorflow-aac568283399
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
import pydicom as dicom
import sys
import cv2
import os
import random
import time
import seaborn
import dcm_contour

np.set_printoptions(threshold=sys.maxsize)

# Constantes utiles pour le whitening
MEAN = -741.7384087183515
STD = 432.83608694943786
MEANXY = 0.5
STDXY = 0.28
MEANZ = 870.6
STDZ = 454.638


def load(path):
    ''' Load the pickle data. The data already contains the 32x32 (32x32x32 in case of 3d) and a correct binary classification'''
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data



def prepareTrainTest_2d(path, strategy):
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
    xy = []
    z = []
    for elem in data:
        # Set the coordonates of the image between 0 and 1
        if strategy == "dense":
            elem[0][1025] = elem[0][1025]/480
            elem[0][1026] = elem[0][1026]/480
            # xy.append(elem[0][1025])
            # xy.append(elem[0][1026])
            # z.append(elem[0][1024])
        # Add the inputs and outputs to the data
        elif strategy == "cnn":
            elem[0] = np.delete(elem[0], [1024, 1025, 1026])
            np.reshape(elem[0], (32, 32))
        inputs.append(elem[0])
        outputs.append(elem[1])
    # xy = np.array(xy)
    # z = np.array(z)
    # print("mean xy: " + str(xy.mean()))
    # print("std xy: " + str(xy.std()))
    # print("mean z: " + str(z.mean()))
    # print("std z: " + str(z.std()))
    outputs = np.asarray(outputs).astype('float32').reshape((-1, 1))
    if strategy == "cnn":
        inputs = np.reshape(inputs, (len(inputs), 32, 32, 1))
    x_train = inputs[:train_size]
    y_train = outputs[:train_size]
    x_test = inputs[train_size:]
    y_test = outputs[train_size:]
    return x_train, y_train, x_test, y_test



def createClients(listdatasetspaths):
    ''' Create dictionary with the clients and their data
        args:
            listdatasetspaths: list of paths to the different detasets
        return:
            clients: dictionary of the different clients and their data. The entries are of the form client_1, client_2 etc
            x_test: test set data coming from the multiple clients
            y_test: test set labels coming from the multiple clients'''
    clients = {}
    x_test = []
    y_test = []
    clientnbr = 0
    for datasetpath in listdatasetspaths:
        x_train_client, y_train_client, x_test_client, y_test_client = prepareTrainTest_2d(datasetpath)
        clientname = "client_" + str(clientnbr)
        data = list(zip(x_train_client, y_train_client))
        clients[clientname] = data
        x_test.extend(x_test_client)
        y_test.extend(y_test_client)
        clientnbr+=1
    return clients, x_test, y_test



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
        elem[0][8192] = elem[0][8192]/600 #600 seems to be a good choice
        elem[0][8193] = elem[0][8193]/480
        elem[0][8194] = elem[0][8194]/480
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



def getZ(dcm_path):
    """ Get the slice_thickness and the initial position of z in order to compute the z position
        args:
            dcm_path: path to the dcm files
        return:
            slice_thickness: the thickness between two slices
    """
    slices = [dicom.dcmread(dcm_path + '/' + s) for s in os.listdir(dcm_path)[1:]]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))  # Sort by z axis
    print(dcm_path)
    try:
        slice_thickness = slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2]
        Z0 = slices[0].ImagePositionPatient[2]
    except:
        slice_thickness = slices[1].SliceLocation - slices[0].SliceLocation
        Z0 = slices[0].SliceLocation
    return slice_thickness, Z0


class SimpleMLP:
    ''' Build and return a simple MLP model'''
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(1000, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(1000))
        model.add(Activation("relu"))
        model.add(Dense(1000))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("sigmoid"))
        return model


class SimpleCNN:
    @staticmethod
    def build(shape, classes):
        ##model building
        model = Sequential()
        # convolutional layer with rectified linear unit activation
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=shape))
        # 32 convolution filters used each of size 3x3
        # again
        model.add(Conv2D(64, (3, 3), activation='relu'))
        # 64 convolution filters used each of size 3x3
        # choose the best features via pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # randomly turn neurons on and off to improve convergence
        model.add(Dropout(0.25))
        # flatten since too many dimensions, we only want a classification output
        model.add(Flatten())
        # fully connected to get all relevant data
        model.add(Dense(128, activation='relu'))
        # one more dropout for convergence' sake :)
        model.add(Dropout(0.5))
        # output a softmax to squash the matrix into output probabilities
        model.add(Dense(classes, activation='sigmoid')),
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


def simpleSGD_2d(x_train, y_train, x_test, y_test, strategy = "dense", lr = 0.01, comms_round = 100):
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
    if strategy == "dense":
        smlp_SGD = SimpleMLP()
        SGD_model = smlp_SGD.build(1027, 1)
    elif strategy == "cnn":
        smlp_SGD = SimpleCNN()
        SGD_model = smlp_SGD.build((32, 32, 1), 1)

    SGD_model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    # fit the SGD training data to model
    _ = SGD_model.fit(SGD_dataset, epochs=100, verbose=2)

    #test the SGD global model and print out metrics

    SGD_acc = test_model(x_test, y_test, SGD_model)

    print("FINAL ACCURACY: " + str(SGD_acc))

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
    SGD_model = smlp_SGD.build(8195, 1)

    SGD_model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    # fit the SGD training data to model
    _ = SGD_model.fit(SGD_dataset, epochs=100, verbose=2)

    SGD_acc = test_model(X_test, y_test, SGD_model)

    print("FINAL ACCURACY: " + str(SGD_acc))

    return SGD_model



def batch_data(data_shard, bs=32):
    ''' Takes in a clients data shard and create a tfds object off it
        args:
            shard: a data, label constituting a client's data shard
            bs:batch size
        return:
            tfds object'''
    # seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)




def weight_scalling_factor(clients_trn_data, client_name, frac):
    ''' Calculates the proportion of a client’s local training data with the overall training data held by all clients
        args:
            clients_trn_data: dictionary of training data by client
            client_name: name of the client
    '''
    client_names = list(clients_trn_data.keys())
    # get the batch size
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    # first calculate the total training data points across clients
    global_count = sum(
        [tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names]) * bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() * bs
    return (local_count / global_count) / frac


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    return avg_grad




def fedAvg(clients, X_test, y_test, frac = 1, bs = 160, epo = 1, lr = 0.01, comms_round = 100):
    ''' federated averaging algorithm
            args:
                clients: dictionary of the clients and their data
                X_test: test set data
                y_test: test set labels
                frac: fraction of clients selected at each round
                bs: local mini-batch size
                epo: number of local epochs
                lr: learning rate
                comms_round: number of global communication round
            returns:
                global_acc: the global accuracy after comms_round rounds
    '''

    # process and batch the training data for each client
    clients_batched = dict()
    for (client_name, data) in clients.items():
        clients_batched[client_name] = batch_data(data, bs = bs)

    # process and batch the test se
    test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))


    loss='binary_crossentropy'
    metrics = ['accuracy']
    optimizer = SGD(lr=lr,
                    decay=lr / comms_round,
                    momentum=0.9
                   )


    # initialize global model
    smlp_global = SimpleMLP()
    global_model = smlp_global.build(1027, 1)


    # commence global training loop
    for comm_round in range(comms_round):

        print('comm_round: ' + str(comm_round))

        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()

        # initial list to collect local model weights after scalling
        scaled_local_weight_list = list()

        # randomize client data - using keys
        client_names = list(clients_batched.keys())
        random.shuffle(client_names)

        # loop through each client and create new local model
        nbrclients = int(frac * len(client_names))
        for client in client_names[:nbrclients]:
            smlp_local = SimpleMLP()
            local_model = smlp_local.build(1027, 1)
            local_model.compile(loss=loss,
                                optimizer=optimizer,
                                metrics=metrics)

            # set local model weight to the weight of the global model
            local_model.set_weights(global_weights)

            # fit local model with client's data
            local_model.fit(clients_batched[client], epochs=epo, verbose=0)

            # scale the model weights and add to list
            scaling_factor = weight_scalling_factor(clients_batched, client, frac)
            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)

            # clear session to free memory after each communication round
            K.clear_session()


        # to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = sum_scaled_weights(scaled_local_weight_list)

        # update global model
        global_model.set_weights(average_weights)

        # test global model and print out metrics after each communications round
        for (X_test, Y_test) in test_batched:
            global_acc = test_model(X_test, Y_test, global_model)

        print('Accuracy after {} rounds = {}'.format(comm_round, global_acc))

    print("FINAL ACCURACY: " + str(global_acc))

    return global_model




def get_dcm_path(client_path):
    dcm_file = os.listdir(client_path)[0]
    dcm_path = client_path + "/" + dcm_file
    dcm_files2 = os.listdir(dcm_path)
    for dcm_file2 in dcm_files2:
        dcm_path2 = dcm_path + "/" + dcm_file2
        if len(os.listdir(dcm_path2)) > 5:
            break
    return dcm_path2


def crop_2d(image, y, x): #vertical, horizontal
    ''' Return a 32x32 image with top left corner of coordonate(y, x)'''
    crop_img = image[y:y + 32, x:x + 32]
    return crop_img



def heatMap(predictions, img_nbr, strategy):
    """ Plot the heat map of the tumor predictions"""
    fig, ax = plt.subplots()
    title = str(strategy) + " model on image: " + str(img_nbr)
    plt.title(title, fontsize=18)
    ttl = ax.title
    ttl.set_position([0.5, 1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    seaborn.heatmap(predictions, ax=ax)
    plt.show()




def finalPrediction(cntr, predictions, threshold):
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            if predictions[(i,j)] > threshold and cntr[(i, j)] != 1:
                cntr[(i, j)] = 0.5
    plt.imshow(cntr)
    plt.show()




def segmentation_2d(model, client_path, img_nbr, speed, strategy):
    ''' process the 2d segmentation of an image and plot the heatmap of the tumor predictions
        args:
            model: model used for predictions
            client_path: path of the client from who the image comes
            img_nbr: number of the image from the patient to be segmented
        return:
            predictions: the 2d array with estimated probability of tumors'''
    array_path = client_path + "/arrays/array_" + str(img_nbr) + ".npy"
    array = np.load(array_path)
    dcm_file0 = os.listdir(client_path)[0]
    dcm_path0 = client_path + "/" + dcm_file0
    dcm_files = os.listdir(dcm_path0)
    for file in dcm_files:
        dcm_path = dcm_path0 + "/" + file
        if len(os.listdir(dcm_path)) > 5:
            break

    # plot the rt struct of the image
    index = dcm_contour.get_index(dcm_path, "GTV-1")
    images, contours = dcm_contour.get_data(dcm_path, index=index)
    for img_arr, contour_arr in zip(images[img_nbr:img_nbr+1], contours[img_nbr:img_nbr+1]):
        dcm_contour.plot2dcontour(img_arr, contour_arr, img_nbr)
    cntr = contours[img_nbr]
    plt.imshow(cntr)
    plt.show()


    predictions = np.zeros((512, 512))
    slice_thickness, Z0 = getZ(dcm_path)
    image = array
    image = image-MEAN
    image = image/STD
    z = Z0 + img_nbr * slice_thickness
    z_whitened = (z - MEANZ) / STDZ
    for y in range(0, 480, speed):
        print("predicting line: " + str(y))
        y_whitened = ((y-MEANXY)/STDXY)/480
        for x in range(0, 480, speed):
            x_whitened = ((x-MEANXY)/STDXY)/480
            if strategy == "dense":
                flatten_subimage = crop_2d(image, y, x).flatten()
                flatten_subimage = np.append(flatten_subimage, z_whitened)
                flatten_subimage = np.append(flatten_subimage, y_whitened)
                flatten_subimage = np.append(flatten_subimage, x_whitened)
                # print(flatten_subimage)
                reshaped = np.reshape(flatten_subimage, (1,1027))
            elif strategy == "cnn":
                subimage = crop_2d(image, y, x)
                reshaped = np.reshape(subimage, (1, 32, 32, 1))
            pred = model.predict(reshaped)
            if pred > 0.5:
                print((pred, y, x))
            # predictions[y:y + 32, x:x + 32] += pred[0]
            predictions[16, 16] += pred[0] #CHECK

    heatMap(predictions, img_nbr, strategy)

    finalPrediction(cntr, predictions, 0.5) #CHECK

    np.save("predictions " + str(strategy), predictions)

    return cntr, predictions



def calibrate(path, client_path, img_nbr, threshold):
    predictions = np.load(path)
    dcm_file0 = os.listdir(client_path)[0]
    dcm_path0 = client_path + "/" + dcm_file0
    dcm_files = os.listdir(dcm_path0)
    for file in dcm_files:
        dcm_path = dcm_path0 + "/" + file
        if len(os.listdir(dcm_path)) > 5:
            break
    # plot the rt struct of the image
    index = dcm_contour.get_index(dcm_path, "GTV-1")
    images, contours = dcm_contour.get_data(dcm_path, index=index)
    cntr = contours[img_nbr]
    finalPrediction(cntr, predictions, threshold)



def crop_3d(image, z, y, x): #vertical, horizontal
    ''' Return a 32x32 image with top left corner of coordonate(y, x) '''
    crop_img = image[z: z + 32, y:y + 32, x:x + 32]
    return crop_img


def segmentation_3d(model, images_path):
    predictions = np.zeros((512, 512, 512))
    image_3d = []
    size = len(os.listdir(images_path))
    for i in range(size):
        array_path = images_path + "/array_" + str(i) +".npy"
        array = np.load(array_path)
        image = array
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
                reshaped = np.reshape(flatten_subimage, (1,8195))
                pred = model.predict(reshaped)
                if pred[0] > 0.5:
                    print("FOUND A TUMOR HERE GUYS")
                    predictions[z:z + 32, y:y + 32, x:x + 32] += 1
    print(predictions)
    return predictions






import pickle
import numpy as np
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import SGD
from keras.models import *
from keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
import dcm_contour
import os
import seaborn
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from keras.layers.advanced_activations import LeakyReLU
import tensorlayer as tl



MEAN = -741.7384087183515
STD = 432.83608694943786


smooth = 1.
# Dice Coefficient to work with Tensorflow
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# Dice Coefficient to work outside Tensorflow
def dice_coef_2(y_true, y_pred):
    side = len(y_true[0])
    y_true_f = y_true.reshape(side*side)
    y_pred_f = y_pred.reshape(side*side)
    intersection = sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)


def load(path):
    ''' Load the pickle data. The data already contains the 32x32 (32x32x32 in case of 3d) and a correct binary classification'''
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data


def prepareTrainTest(dataset_path):
    dataset = load(dataset_path)
    dataset = np.array(dataset)
    np.random.shuffle(dataset)
    size = len(dataset)
    train_size = int(0.8 * size)
    inputs = []
    outputs = []
    for elem in dataset:
        inputs.append(elem[0])
        outputs.append(elem[1])
    x_train = inputs[:train_size]
    y_train = outputs[:train_size]
    x_test = inputs[train_size:]
    y_test = outputs[train_size:]
    return x_train, y_train, x_test, y_test


def get_model2(optimizer, loss_metric, metrics, lr=1e-3):
    inputs = Input((512, 512, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.5)(pool1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(drop1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(drop2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.3)(pool3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(drop3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop4 = Dropout(0.3)(pool4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(drop4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=optimizer(lr=lr), loss=loss_metric, metrics=metrics)

    return model


def simpleSGD_2d(x_train, y_train, x_test, y_test):
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

    # unet = Unet()
    # model = unet.initial_model()
    # model.summary()

    # checkpointer = ModelCheckpoint('unet_model_Lungs_first5_temp.h5', verbose=1, save_best_only=True)
    #
    # callbacks = [
    #     EarlyStopping(patience=3, monitor='val_loss'),
    #     TensorBoard(log_dir='logs'),
    #     checkpointer
    # ]

    model = get_model2(optimizer=tf.keras.optimizers.Adam, loss_metric=dice_coef_loss, metrics=[dice_coef, "accuracy"], lr=1e-4)

    # model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

    model.summary()

    # Train the model, doing validation at the end of each epoch.
    epochs = 50 #CHECK
    model.fit(x_train, y_train, batch_size=16, epochs=epochs) # CHECK callbacks


    return model



def heatMap(predictions):
    """ Plot the heat map of the tumor predictions"""
    fig, ax = plt.subplots()
    title = "Prediction heatmap"
    plt.title(title, fontsize=18)
    ttl = ax.title
    ttl.set_position([0.5, 1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    seaborn.heatmap(predictions, ax=ax)
    plt.show()




def finalPrediction(cntr, predictions, zone):
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            if predictions[(i, j)] >= 0.5 and cntr[(i, j)] != 1:
                cntr[(i, j)] = 10
    title = zone + " prediction"
    plt.title(title, fontsize=18)
    plt.imshow(cntr)
    plt.show()



def segmentation_2d(model, client_path, img_nbr, zone):
    ''' process the 2d segmentation of an image and plot the heatmap of the tumor predictions
        args:
            model: model used for predictions
            client_path: path of the client from who the image comes
            img_nbr: number of the image from the patient to be segmented
        return:
            predictions: the 2d array with estimated probability of tumors'''
    dcm_file0 = os.listdir(client_path)[0]
    dcm_path0 = client_path + "/" + dcm_file0
    dcm_files = os.listdir(dcm_path0)
    for file in dcm_files:
        dcm_path = dcm_path0 + "/" + file
        if len(os.listdir(dcm_path)) > 5:
            break

    if zone == "tumor":
        # plot the rt struct of the image
        index = dcm_contour.get_index(dcm_path, "GTV-1")
        images, contours, _ = dcm_contour.get_data(dcm_path, index=index)
        for img_arr, contour_arr in zip(images[img_nbr:img_nbr+1], contours[img_nbr:img_nbr+1]):
            dcm_contour.plot2dcontour(img_arr, contour_arr, img_nbr)
        cntr = contours[img_nbr]
    elif zone == "lungs":
        index1 = dcm_contour.get_index(dcm_path, "Lung-Left")
        index2 = dcm_contour.get_index(dcm_path, "Lung-Right")
        images, contours1, _ = dcm_contour.get_data(dcm_path, index=index1)
        images2, contours2, _ = dcm_contour.get_data(dcm_path, index=index2)
        for i in range(len(contours1)):
            contours1[i] = np.append(contours1[i], contours2[i], axis=0)
        for img_arr, contour_arr in zip(images[img_nbr:img_nbr + 1], contours1[img_nbr:img_nbr + 1]):
            dcm_contour.plot2dcontour(img_arr, contour_arr, img_nbr)
        cntr = contours1[img_nbr][0]
        for i in range(1, len(contours1[img_nbr])):
            print("ici")
            cntr += contours1[img_nbr][i]
    plt.show()

    array_path = client_path + "/arrays/array_" + str(img_nbr) + ".npy"
    array = np.load(array_path)
    array = array - MEAN
    array = array / STD
    array = np.reshape(array, (1, 512, 512, 1))
    predictions = model.predict(array)
    predictions = np.reshape(predictions, (512, 512))

    heatMap(predictions)

    finalPrediction(cntr, predictions, zone)

    if zone == "tumor":
        mask_path = client_path + "/masks/mask_" + str(img_nbr) + ".png"
    elif zone == "lungs":
        mask_path = client_path + "/masks_Lungs/mask_" + str(img_nbr) + ".png"
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask[mask < 40] = 0  # Set out of tumor to 0
    mask[mask > 210] = 1  # Set out of tumor to 1
    dice = dice_coef_2(mask, predictions)
    print("dice accuracy: " + str(dice))

    return cntr, predictions
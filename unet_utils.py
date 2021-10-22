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
from tensorflow.keras import backend as K
import tensorflow as tf
import dcm_contour
import os
import seaborn
import matplotlib.pyplot as plt
import cv2


MEAN = -741.7384087183515
STD = 432.83608694943786


def SDC(TP, FP, FN):
    return 2*TP/(2*TP + FP + FN)



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



class Unet():
    def __init__(self):
        print("Initial U-Net model...")
        self.model = self.initial_model()

    def initial_model(self):
        concat_axis = 3

        inputs = Input((512, 512, 1))
        conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv1_1')(inputs)
        conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv1_2')(conv1_1)
        pool1 = MaxPooling2D(pool_size=(2, 2), name='maxpooling_1')(conv1_2)

        conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='valid', name='conv2_1')(pool1)
        conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='valid', name='conv2_2')(conv2_1)
        pool2 = MaxPooling2D(pool_size=(2, 2), name='maxpooling_2')(conv2_2)

        conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv3_1')(pool2)
        conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv3_2')(conv3_1)
        pool3 = MaxPooling2D(pool_size=(2, 2), name='maxpooling_3')(conv3_2)

        conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='valid', name='conv4_1')(pool3)
        conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='valid', name='conv4_2')(conv4_1)
        pool4 = MaxPooling2D(pool_size=(2, 2), name='maxpooling_4')(conv4_2)

        conv5_1 = Conv2D(1024, (3, 3), activation='relu', padding='valid', name='conv5_1')(pool4)
        conv5_2 = Conv2D(1024, (3, 3), activation='relu', padding='valid', name='conv5_2')(conv5_1)

        upsampling1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='valid', name='upsampling1')(conv5_2)
        crop_conv4_2 = Cropping2D(cropping=((4, 4), (4, 4)), name='cropped_conv4_2')(conv4_2)
        up6 = concatenate([upsampling1, crop_conv4_2], axis=concat_axis, name='skip_connection1')
        conv6_1 = Conv2D(512, (3, 3), activation='relu', padding='valid', name='conv6_1')(up6)
        conv6_2 = Conv2D(512, (3, 3), activation='relu', padding='valid', name='conv6_2')(conv6_1)

        upsampling2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='valid', name='upsampling2')(conv6_2)
        crop_conv3_2 = Cropping2D(cropping=((16, 16), (16, 16)), name='cropped_conv3_2')(conv3_2)
        up7 = concatenate([upsampling2, crop_conv3_2], axis=concat_axis, name='skip_connection2')
        conv7_1 = Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv7_1')(up7)
        conv7_2 = Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv7_2')(conv7_1)

        upsampling3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='valid', name='upsampling3')(conv7_2)
        crop_conv2_2 = Cropping2D(cropping=((40, 40), (40, 40)), name='cropped_conv2_2')(conv2_2)
        up8 = concatenate([upsampling3, crop_conv2_2], axis=concat_axis, name='skip_connection3')
        conv8_1 = Conv2D(128, (3, 3), activation='relu', padding='valid', name='conv8_1')(up8)
        conv8_2 = Conv2D(128, (3, 3), activation='relu', padding='valid', name='conv8_2')(conv8_1)

        upsampling4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='valid', name='upsampling4')(conv8_2)
        crop_conv1_2 = Cropping2D(cropping=((88, 88), (88, 88)), name='cropped_conv1_2')(conv1_2)
        up9 = concatenate([upsampling4, crop_conv1_2], axis=concat_axis, name='skip_connection4')
        conv9_1 = Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv9_1')(up9)
        conv9_2 = Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv9_2')(conv9_1)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='conv10')(conv9_2)

        model = Model(inputs=[inputs], outputs=[conv10])
        return model

    def model_summary(self):
        self.model.summary()

    def get_model(self):
        return self.model


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (1,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model



def test_model(x_test, y_test, model):
    """ Calculates the accuracy and the loss of the model
        args:
            x_test: test set data
            y_test: test set labels
            model: the model
        return:
            acc: accuracy of the model

    """
    predictions = []
    accs = []
    y_test = np.array(y_test)
    for k in range(len(x_test)):
        x_test[k] = np.reshape(x_test[k], (1, 512, 512, 1))
        prediction = model.predict(x_test[k])
        prediction = np.reshape(prediction, (512, 512))
        predictions.append(prediction)
    predictions = np.array(predictions)
    for w in range(len(predictions)):
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                if predictions[(w, i, j)] > 0.5:
                    predictions[(w, i, j)] = 1
                else:
                    predictions[(w, i, j)] = 0
        TP = 0
        FP = 0
        FN = 0
        for k in range(len(predictions[w])):
            for k2 in range(len(predictions[(w, k)])):
                if predictions[(w, k, k2)] == 0:
                    if y_test[(w, k, k2)] == 1:
                        FN += 1
                if predictions[(w, k, k2)] == 1:
                    if y_test[(w, k, k2)] == 1:
                        TP += 1
                    else:
                        FP += 1
        acc = SDC(TP, FP, FN)
        accs.append(acc)
    print("Finale accuracy = " + str(acc))
    return acc



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

    model = get_model((512, 512), 1)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    checkpointer = ModelCheckpoint("U-net_model.h5", verbose=1, save_best_only=True)

    callbacks = [
        EarlyStopping(patience=3, monitor='val_loss'),
        TensorBoard(log_dir='logs'),
        checkpointer
    ]

    # Train the model, doing validation at the end of each epoch.
    epochs = 15
    model.fit(x_train, y_train, batch_size=16, epochs=epochs, callbacks=callbacks)

    test_model(x_test, y_test, model)


    return model



def heatMap(predictions, img_nbr):
    """ Plot the heat map of the tumor predictions"""
    fig, ax = plt.subplots()
    title = " model on image: " + str(img_nbr)
    plt.title(title, fontsize=18)
    ttl = ax.title
    ttl.set_position([0.5, 1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    seaborn.heatmap(predictions, ax=ax)
    plt.show()




def finalPrediction(cntr, predictions):
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            if predictions[(i,j)] > 0.5 and cntr[(i, j)] != 1:
                cntr[(i, j)] = 0.5
    plt.imshow(cntr)
    plt.show()



def segmentation_2d(model, client_path, img_nbr):
    ''' process the 2d segmentation of an image and plot the heatmap of the tumor predictions
        args:
            model: model used for predictions
            client_path: path of the client from who the image comes
            img_nbr: number of the image from the patient to be segmented
        return:
            predictions: the 2d array with estimated probability of tumors'''
    array_path = client_path + "/arrays/array_" + str(img_nbr) + ".npy"
    array = np.load(array_path)
    array = array - MEAN
    array = array / STD
    dcm_file0 = os.listdir(client_path)[0]
    dcm_path0 = client_path + "/" + dcm_file0
    dcm_files = os.listdir(dcm_path0)
    for file in dcm_files:
        dcm_path = dcm_path0 + "/" + file
        if len(os.listdir(dcm_path)) > 5:
            break
    mask_file = client_path + "/masks/mask_" + str(img_nbr) + ".png"
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    if np.sum(mask) > 7864320:  # If there is a tumor
        mask[mask < 40] = 0  # Set out of tumor to 0
        mask[mask > 210] = 1  # Set out of tumor to 1

    # plot the rt struct of the image
    index = dcm_contour.get_index(dcm_path, "GTV-1")
    images, contours = dcm_contour.get_data(dcm_path, index=index)
    for img_arr, contour_arr in zip(images[img_nbr:img_nbr+1], contours[img_nbr:img_nbr+1]):
        dcm_contour.plot2dcontour(img_arr, contour_arr, img_nbr)
    cntr = contours[img_nbr]
    plt.imshow(cntr)
    plt.show()



    array = np.reshape(array, (1, 512, 512, 1))
    predictions = model.predict(array)
    predictions = np.reshape(predictions, (512, 512))

    heatMap(predictions, img_nbr)

    finalPrediction(cntr, predictions)

    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            if predictions[(i, j)] > 0.5:
                predictions[(i, j)] = 1
            else:
                predictions[(i, j)] = 0
    TP = 0
    FP = 0
    FN = 0
    for k in range(len(predictions)):
        for k2 in range(len(predictions[k])):
            if predictions[(k, k2)] == 0:
                if mask[(k, k2)] == 1:
                    FN += 1
            if predictions[(k, k2)] == 1:
                if mask[(k, k2)] == 1:
                    TP += 1
                else:
                    FP += 1
    acc = SDC(TP, FP, FN)
    print("Finale accuracy = " + str(acc))



    np.save("predictions " + str(img_nbr), predictions)

    return cntr, predictions
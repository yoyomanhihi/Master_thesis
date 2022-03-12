import pickle

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import History
from tensorflow.keras import backend as K
import tensorflow as tf
import os
import cv2
import random
import plots
import imageio
import file_utils

MEAN = 4611.838943481445
STD = 7182.589254997573


def get_mean_std(images_path):
    means = []
    stds = []
    images = os.listdir(images_path)
    for name in images:
        image_file = images_path + '/' + name
        image = imageio.imread(image_file)
        image[0][0] = 0
        mean = np.mean(image)
        std = np.std(image)
        means.append(mean)
        stds.append(std)
        print('mean: ' + str(mean))
        print('std: ' + str(std))
    return np.mean(means), np.mean(std)

smooth = 5000. #tocheck
# Dice Coefficient to work with Tensorflow
def dice_coef_ponderated(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.sum(y_true) * ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


def dice_coef_loss_ponderated(y_true, y_pred):
    return -dice_coef_ponderated(y_true, y_pred)


smoother = 5000.
# Dice Coefficient to work with Tensorflow
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_small_smooth(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def no_organ_coef(y_pred, smoother):
    y_pred_f = K.flatten(y_pred)
    error = smoother + K.sum(y_pred_f)
    return smoother/error


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def dice_coef_loss_custom(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    sum = K.sum(y_true_f)
    if sum > 0:
        return -dice_coef(y_true, y_pred)
    else:
        return -no_organ_coef(y_pred, smoother)


def get_average_number_of_true_pixels(datasetpath):
    true_pixels = 0
    masks_path = datasetpath + '/test/masks'
    for mask_file in os.listdir(masks_path):
        mask_path = masks_path + '/' + mask_file
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        true_pixels += np.sum(mask)
    return true_pixels / len(os.listdir(masks_path))


# Dice Coefficient to work outside Tensorflow
def dice_coef_2(y_true, y_pred, smooth):
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


def get_model():
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

    return model


def test_model(datasetpath, model):
    test_generator = dataAugmentation(datasetpath, class_train='test')
    len_test = len(os.listdir(datasetpath + '/test/images'))
    test_acc = model.evaluate_generator(generator=test_generator, steps=len_test / 1)
    return test_acc


def dice_3d(datasetpath, model, i):
    test_path = datasetpath + '/test'
    images_path = os.listdir(test_path + '/images')
    masks_path = os.listdir(test_path + '/masks')
    image = imageio.imread(test_path + '/images/' + images_path[i])
    image[0][0] = 0
    image = image - MEAN
    image = image / STD
    mask = cv2.imread(test_path + '/masks/' + masks_path[i], cv2.IMREAD_GRAYSCALE)
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    image = np.reshape(image, (1, 512, 512, 1))
    prediction = model.predict(image)
    prediction[prediction > 0.5] = 1
    prediction[prediction <= 0.5] = 0
    side = len(mask[0])
    prediction = prediction.reshape(512, 512)
    # plt.imshow(prediction)
    # plt.show()
    y_true_f = mask.reshape(side * side)
    y_pred_f = prediction.reshape(side * side)
    intersection = sum(y_true_f * y_pred_f)
    sum_of_true = sum(y_true_f)
    sum_of_true += sum(y_pred_f)
    return intersection, sum_of_true


def test_model_3d(datasetpath, model):
    test_path = datasetpath + '/test'
    model = tf.keras.models.load_model(model, compile=False)
    patient_intersection = 0
    patient_sumtrue = 0
    last_patient = 0
    dices = []
    beginning = True
    len_data = len(os.listdir(test_path + '/images'))
    for i in range(len_data):
        images_path = os.listdir(test_path + '/images')
        patient_nbr = int(images_path[i].split('_')[0])
        if beginning:
            beginning = False
            intersection, sum_of_true = dice_3d(datasetpath, model, i)
            patient_intersection = intersection
            patient_sumtrue = sum_of_true
            last_patient = patient_nbr
        elif (patient_nbr == last_patient and i < len_data-1):
            intersection, sum_of_true = dice_3d(datasetpath, model, i)
            patient_intersection += intersection
            patient_sumtrue += sum_of_true
        elif i == len_data-1:
            intersection, sum_of_true = dice_3d(datasetpath, model, i)
            patient_intersection += intersection
            patient_sumtrue += sum_of_true
            dice = (2 * patient_intersection) / (patient_sumtrue)
            dices.append(dice)
        else:
            dice = (2 * patient_intersection) / (patient_sumtrue)
            dices.append(dice)
            intersection, sum_of_true = dice_3d(datasetpath, model, i)
            patient_intersection = intersection
            patient_sumtrue = sum_of_true
            last_patient = patient_nbr

    print(dices)
    return np.mean(dices)


def adjustData(img, mask, class_train):
    for image in img:
        image[0][0] = 0
        # print(np.max(image))
        # plt.imshow(image, cmap='gray', vmin=0, vmax=65535)
        # plt.show()

    # Random brightness change
    if class_train == 'train':
        brightness = random.uniform(0.985, 1.015) # tocheck
        for i in range(len(img)):
            img[i] = img[i] * brightness

    img = img-MEAN
    img = img/STD
    mask = mask / 255
    # plt.imshow(img[0], cmap='gray', interpolation='none')
    # plt.show()
    # plt.imshow(mask[0], cmap='gray', interpolation='none')
    # plt.show()
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img,mask)


def dataAugmentation(train_data_dir, class_train = 'train'):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''

    if(class_train == 'train'):

        image_datagen = ImageDataGenerator(dtype=tf.uint16, zoom_range=0.08, rotation_range=25) # tocheck
        mask_datagen = ImageDataGenerator(dtype=tf.uint16, zoom_range=0.08, rotation_range=25)

        image_generator = image_datagen.flow_from_directory(
            train_data_dir + '/' + class_train,
            classes=['images'],
            class_mode=None,
            color_mode='grayscale',
            target_size=(512, 512),
            batch_size=1,
            shuffle=True,
            seed=1)
        mask_generator = mask_datagen.flow_from_directory(
            train_data_dir + '/' + class_train,
            classes=['masks'],
            class_mode=None,
            color_mode='grayscale',
            target_size=(512, 512),
            batch_size=1,
            shuffle=True,
            seed=1)

    elif class_train == 'validation' or class_train == 'test':

        image_datagen = ImageDataGenerator(dtype=tf.uint16)
        mask_datagen = ImageDataGenerator(dtype=tf.uint16)

        image_generator = image_datagen.flow_from_directory(
            train_data_dir + '/' + class_train,
            classes=['images'],
            class_mode=None,
            color_mode='grayscale',
            target_size=(512, 512),
            batch_size=1,
            shuffle=False,
            seed=1)
        mask_generator = mask_datagen.flow_from_directory(
            train_data_dir + '/' + class_train,
            classes=['masks'],
            class_mode=None,
            color_mode='grayscale',
            target_size=(512, 512),
            batch_size=1,
            shuffle=False,
            seed=1)


    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, class_train)
        yield (img, mask)

    return train_generator


def simpleSGD(datasetpath, preloaded, epochs, name):
    ''' Simple SGD algorithm for 32x32 images
        args:
            x_train: training images
            y_train: training labels
            x_test: test images
            y_test: test labelsValueError: Layer model expects 1 input(s), but it received 90 input tensors
            lr: learning rate
            comms_round: number of communication rounds
        returns:
            SGD_model: model fitted
    '''

    model_name = name + '.h5'

    checkpointer = ModelCheckpoint(model_name, verbose=1, save_best_only=True)

    history = History()

    callbacks = [
        EarlyStopping(patience=5, monitor='val_loss', mode=min), #tocheck
        TensorBoard(log_dir='logs'),
        history,
        checkpointer,
    ]

    optimizer = tf.keras.optimizers.Adam
    loss_metric = dice_coef_loss # tocheck
    metrics = [dice_coef]
    # lr = lr_scheduler.TanhDecayScheduler()
    lr = 5e-5 # tocheck

    model = get_model()

    if preloaded is not None:
        model.load_weights(preloaded)

    model.compile(optimizer=optimizer(learning_rate=lr), loss=loss_metric, metrics=metrics)

    training_generator = dataAugmentation(datasetpath, class_train='train')
    validation_generator = dataAugmentation(datasetpath, class_train='validation')

    len_training = len(os.listdir(datasetpath + '/train/images'))
    len_validation = len(os.listdir(datasetpath + '/validation/images'))

    hist = model.fit(training_generator, validation_data=validation_generator, validation_steps=len_validation/1, steps_per_epoch=len_training/1, epochs=epochs, shuffle=True, callbacks=callbacks, verbose=1) # tocheck len attention

    # Train the model, doing validation at the end of each epoch.
    # hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), shuffle=True, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    print((hist.history['val_dice_coef']))

    measures_file = name + "_data.txt"

    train_accs = hist.history['dice_coef']
    val_accs = hist.history['val_dice_coef']

    file_utils.write_measures(measures_file, train_accs)
    file_utils.write_measures(measures_file, val_accs)

    plots.history(train_accs, val_accs, name)
    return model


def get_ratio_of_clients(nbrclients, datasetpath):
    ''' Calculates the proportion of a client’s local training data with the overall training data held by all clients
        args:
            nbrclients: number of clients
            datasetpath: path to the dataset with all clients
    '''
    sizes = []
    totalsize = 0

    for i in range(nbrclients):
        client_path = datasetpath + '/' + str(i) + "/train/images"
        client_size = len(os.listdir(client_path))
        sizes.append(client_size)
        totalsize += client_size

    sizes = np.array(sizes)
    return sizes/totalsize


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



def fedAvg_2(datasetpath, preloaded, nbrclients, name, frac = 1, epo = 1, comms_round = 100, patience = 5): # tocheck
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

    model_name = name + '.h5'

    best_val_acc = 0.
    val_accs = []
    train_acc = []
    global_val_accs = []
    patience_wait = 0

    clients_weight = get_ratio_of_clients(nbrclients, datasetpath)

    print("clients_weight: " + str(clients_weight))

    # process and batch the training data for each client
    # clients_batched = dict()
    # for (client_name, data) in clients.items():
    #     clients_batched[client_name] = batch_data(data, bs = bs)

    # validation_generator = dataAugmentation(datasetpath, class_train='validation')

    history = History()

    callbacks = [
        history,
    ]

    optimizer = tf.keras.optimizers.Adam
    loss_metric = dice_coef_loss
    metrics = [dice_coef]
    lr = 5e-5

    # initialize global model
    global_model = get_model()
    if preloaded is not None:
        global_model.load_weights(preloaded)

    global_model.compile(optimizer=optimizer(learning_rate=lr), loss=loss_metric, metrics=metrics)


    # start global training loop
    for comm_round in range(comms_round):

        print('comm_round: ' + str(comm_round))

        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()

        # initial list to collect local model weights after scalling
        scaled_local_weight_list = list()

        clients = list(range(0, nbrclients))

        # random.shuffle(clients)
        print('clients: ' + str(clients))

        # loop through each client and create new local model
        nbrclients = int(frac * nbrclients)
        # for client in client_names[:nbrclients]:
        for client in clients[:nbrclients]:

            client_path = datasetpath + '/' + str(client)

            local_model = get_model()

            local_model.compile(optimizer=optimizer(learning_rate=lr), loss=loss_metric, metrics=metrics)

            # set local model weight to the weight of the global model
            local_model.set_weights(global_weights)

            training_generator = dataAugmentation(client_path, class_train='train')

            len_training = len(os.listdir(client_path + '/train/images'))

            train_acc = local_model.fit(training_generator,
                                       steps_per_epoch=len_training*(clients_weight[1]/clients_weight[client]), epochs=epo, shuffle=True,
                                       callbacks=callbacks, verbose=1)

            # scale the model weights and add to list

            # scaling_factor = weight_scaling_factor(clients_batched, client, frac)

            # scaling_factor = clients_weight[client]
            scaling_factor = 1./nbrclients # Because small datasets will be train longer

            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)

            # clear session to free memory after each communication round
            K.clear_session()

        # to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = sum_scaled_weights(scaled_local_weight_list)

        # update global model
        global_model.set_weights(average_weights)

        # process and batch the test se
        # test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

        # test global model and print out metrics after each communications round
        # for (X_test, Y_test) in test_batched:

        mean_val_acc = 0.

        for client in clients[:nbrclients]:
            client_path = datasetpath + '/' + str(client)
            validation_generator = dataAugmentation(client_path, class_train='validation')
            len_validation = len(os.listdir(client_path + '/validation/images'))
            val_acc = global_model.evaluate_generator(generator=validation_generator, steps=len_validation/1)

            val_acc=val_acc[1] # Dice's coeff

            mean_val_acc+=val_acc

            val_accs.append(val_acc)

        mean_val_acc = mean_val_acc / nbrclients

        # If best acc so far
        if mean_val_acc > best_val_acc:
            print("Average validation Dice score improved: from " + str(best_val_acc) + " to: " + str(mean_val_acc))
            best_val_acc = mean_val_acc
            patience_wait = 0
            global_model.save(model_name)
        else:
            print("Average validation Dice score didn't improve, got a dice score of " + str(mean_val_acc) + " but best is still " + str(best_val_acc) )
            patience_wait += 1
        print("patience_wait = " + str(patience_wait))

        # Stores global_validation_acc
        global_val_path = "datasets/dataset_heart" # tocheck
        validation_generator = dataAugmentation(global_val_path, class_train='validation')
        len_validation = len(os.listdir(global_val_path + '/validation/images'))
        global_val_acc = global_model.evaluate_generator(generator=validation_generator, steps=len_validation / 1)
        global_acc = global_val_acc[1]
        global_val_accs.append(global_acc)

        if patience_wait >= patience:
            break

        print('Accuracy after {} rounds = {}'.format(comm_round, mean_val_acc))

    print("FINAL ACCURACY: " + str(mean_val_acc))

    train_accs = train_acc.history['dice_coef']

    measures_file = name + "_data.txt"

    file_utils.write_measures(measures_file, train_accs)
    file_utils.write_measures(measures_file, val_accs)
    file_utils.write_measures(measures_file, global_val_accs)

    plots.history_fedavg(train_accs, val_accs, len(clients), name)

    return global_model


def fedAvg_original(datasetpath, preloaded, nbrclients, name, frac = 1, epo = 1, comms_round = 100, patience = 5): # tocheck
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

    model_name = name + '.h5'

    best_val_acc = 0.
    val_accs = []
    train_acc = []
    global_val_accs = []
    patience_wait = 0

    clients_weight = get_ratio_of_clients(nbrclients, datasetpath)

    print("clients_weight: " + str(clients_weight))

    # process and batch the training data for each client
    # clients_batched = dict()
    # for (client_name, data) in clients.items():
    #     clients_batched[client_name] = batch_data(data, bs = bs)

    # validation_generator = dataAugmentation(datasetpath, class_train='validation')

    history = History()

    callbacks = [
        history,
    ]

    optimizer = tf.keras.optimizers.Adam
    loss_metric = dice_coef_loss
    metrics = [dice_coef]
    lr = 5e-5

    # initialize global model
    global_model = get_model()
    if preloaded is not None:
        global_model.load_weights(preloaded)

    global_model.compile(optimizer=optimizer(learning_rate=lr), loss=loss_metric, metrics=metrics)


    # start global training loop
    for comm_round in range(comms_round):

        print('comm_round: ' + str(comm_round))

        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()

        # initial list to collect local model weights after scalling
        scaled_local_weight_list = list()

        # # randomize client data - using keys
        # client_names = list(clients_batched.keys())
        # random.shuffle(client_names)

        clients = list(range(0, nbrclients))

        # random.shuffle(clients)
        print('clients: ' + str(clients))

        # loop through each client and create new local model
        nbrclients = int(frac * nbrclients)
        # for client in client_names[:nbrclients]:
        for client in clients[:nbrclients]:

            client_path = datasetpath + '/' + str(client)

            local_model = get_model()

            local_model.compile(optimizer=optimizer(learning_rate=lr), loss=loss_metric, metrics=metrics)

            # set local model weight to the weight of the global model
            local_model.set_weights(global_weights)

            training_generator = dataAugmentation(client_path, class_train='train')

            len_training = len(os.listdir(client_path + '/train/images'))

            train_acc = local_model.fit(training_generator,
                                       steps_per_epoch=len_training, epochs=epo, shuffle=True,
                                       callbacks=callbacks, verbose=1)

            # scale the model weights and add to list

            scaling_factor = clients_weight[client]

            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)

            # clear session to free memory after each communication round
            K.clear_session()

        # to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = sum_scaled_weights(scaled_local_weight_list)

        # update global model
        global_model.set_weights(average_weights)

        # process and batch the test se
        # test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

        # test global model and print out metrics after each communications round
        # for (X_test, Y_test) in test_batched:

        mean_val_acc = 0.

        for client in clients[:nbrclients]:
            client_path = datasetpath + '/' + str(client)
            validation_generator = dataAugmentation(client_path, class_train='validation')
            len_validation = len(os.listdir(client_path + '/validation/images'))
            val_acc = global_model.evaluate_generator(generator=validation_generator, steps=len_validation/1)

            val_acc=val_acc[1] # Dice's coeff

            mean_val_acc+=val_acc

            val_accs.append(val_acc)

        mean_val_acc = mean_val_acc / nbrclients

        # If best acc so far
        if mean_val_acc > best_val_acc:
            print("Average validation Dice score improved: from " + str(best_val_acc) + " to: " + str(mean_val_acc))
            best_val_acc = mean_val_acc
            patience_wait = 0
            global_model.save(model_name)
        else:
            print("Average validation Dice score didn't improve, got a dice score of " + str(mean_val_acc) + " but best is still " + str(best_val_acc) )
            patience_wait += 1
        print("patience_wait = " + str(patience_wait))

        # Stores global_validation_acc
        global_val_path = "datasets/dataset_heart" # tocheck
        validation_generator = dataAugmentation(global_val_path, class_train='validation')
        len_validation = len(os.listdir(global_val_path + '/validation/images'))
        global_val_acc = global_model.evaluate_generator(generator=validation_generator, steps=len_validation / 1)
        global_acc = global_val_acc[1]
        global_val_accs.append(global_acc)

        if patience_wait >= patience:
            break

        print('Accuracy after {} rounds = {}'.format(comm_round, mean_val_acc))

    print("FINAL ACCURACY: " + str(mean_val_acc))

    train_accs = train_acc.history['dice_coef']

    measures_file = name + "_data.txt"

    file_utils.write_measures(measures_file, train_accs)
    file_utils.write_measures(measures_file, val_accs)
    file_utils.write_measures(measures_file, global_val_accs)

    plots.history_fedavg(train_accs, val_accs, len(clients), name)

    return global_model




'''
def prepareTrainingData(dataset_path):
    dataset = load(dataset_path)
    dataset = np.array(dataset)
    x_train = []
    y_train = []
    for elem in dataset:
        x_train.append(elem[0])
        y_train.append(elem[1])
    x_train = np.reshape(x_train, (len(x_train), 512, 512, 1))
    y_train = np.reshape(y_train, (len(y_train), 512, 512, 1))
    return x_train, y_train


def prepareTrainTest(dataset_path):
    dataset = load(dataset_path)
    dataset = np.array(dataset)
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
'''

'''
def createClients(listdatasetspaths):
     Create dictionary with the clients and their data
        args:
            listdatasetspaths: list of paths to the different detasets
        return:
            clients: dictionary of the different clients and their data. The entries are of the form client_1, client_2 etc
            x_test: test set data coming from the multiple clients
            y_test: test set labels coming from the multiple clients
    clients = {}
    x_test = []
    y_test = []
    clientnbr = 0
    for datasetpath in listdatasetspaths:
        x_train_client, y_train_client, x_test_client, y_test_client = prepareTrainTest(datasetpath)
        clientname = "client_" + str(clientnbr)
        data = list(zip(x_train_client, y_train_client))
        clients[clientname] = data
        x_test.extend(x_test_client)
        y_test.extend(y_test_client)
        clientnbr+=1
    return clients, x_test, y_test
    
    
def batch_data(data_shard, bs=3):
    Takes in a clients data shard and create a tfds object off it
        args:
            shard: a data, label constituting a client's data shard
            bs:batch size
        return:
            tfds object
    # seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)
    
def weight_scaling_factor(clients_trn_data, client_name, frac):
    Calculates the proportion of a client’s local training data with the overall training data held by all clients
        args:
            clients_trn_data: dictionary of training data by client
            client_name: name of the client
    
    client_names = list(clients_trn_data.keys())
    # get the batch size
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    # first calculate the total training data points across clients
    global_count = sum(
        [tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names]) * bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() * bs
    return (local_count / global_count) / frac
'''

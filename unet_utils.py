# Federated learning functions inspired by the great tutorial available here:
# https://towardsdatascience.com/federated-learning-a-step-by-step-implementation-in-tensorflow-aac568283399
#
# U-Net model inspired by another great tutorial that can be found here:
# https://medium.com/@fabio.sancinetti/u-net-convnet-for-ct-scan-segmentation-6cc0d465eed3

import unet_preprocessing
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
import unet_segmentation

# Mean and std of the pixels in the dataset
MEAN = 4611.838943481445
STD = 7182.589254997573

def get_mean_std(images_path):
    """ Compute the mean and std of the pixels in the dataset
        Used to compute MEAN and STD defined above
    Args:
        images_path: path to the images
    """
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

# value of the smooth term used in the loss function
smooth = 1. #tocheck

# Dice Coefficient to work with Tensorflow
def dice_coef(y_true, y_pred):
    """ Dice's coefficient
    Args:
        y_true: correct mask of the organ
        y_pred: predicted mask of the organ
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    """ Dice's loss
    Args:
        y_true: correct mask of the organ
        y_pred: predicted mask of the organ
    """
    return -dice_coef(y_true, y_pred)


def get_average_number_of_true_pixels(datasetpath):
    """ Compute the average number of true pixels of the images
        in the dataset. Used for a custom loss function
    Args:
        datasetpath: path to the dataset
    """
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


def jaccard_distance(y_true, y_pred):
    """ Jaccard distance
    Args:
        y_true: correct mask of the organ
        y_pred: predicted mask of the organ
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def dice_coef_2(y_true, y_pred, smooth):
    """ Dice's coefficient to work outside Tensorflow
    Args:
        y_true: correct mask of the organ
        y_pred: predicted mask of the organ
        smooth: value of the smooth term
    """
    side = len(y_true[0])
    y_true_f = y_true.reshape(side*side)
    y_pred_f = y_pred.reshape(side*side)
    intersection = sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)


def get_model():
    """ Create the U-net model
    Return:
        model: the model
    """
    inputs = Input((512, 512, 1)) # size of the images

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs) # 3x3 convolution
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)  # 3x3 convolution
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # max pooling layer
    drop1 = Dropout(0.5)(pool1) # dropout layer

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

    up9 = concatenate([Conv2DTranspose(32, (2, 2),
                                       strides=(2, 2),
                                       padding='same')
                       (conv8), conv1], axis=3)         # skip connections

    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)   # 3x3 convolution
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9) # 3x3 convolution

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9) # 1x1 sigmoid activation

    model = Model(inputs=[inputs], outputs=[conv10]) # define the model

    return model


def test_model(datasetpath, model):
    """ Test the model on the test set
    Args:
        datasetpath: path to the test set
        model: the model
    Return:
        test_acc: the score of the model on the test set
    """

    # Create a generetor of images for the test set (without data augmentation)
    test_generator = dataAugmentation(datasetpath, class_train='test')
    len_test = len(os.listdir(datasetpath + '/test/images'))
    test_acc = model.evaluate_generator(generator=test_generator, steps=len_test / 1)
    return test_acc


def count_prediction(datasetpath, model, i):
    """ count the number of true positive and sum of the positive
        Useful to compute the 3d Dice's coefficient
    Args:
        datasetpath: path to the test set
        model: the model
        i: the number of the image to predict
    Return:
        intersection: the number of true positive
        sum_of_true: the sum of the positive pixels from the real and predicted masks
    """

    test_path = datasetpath + '/test'
    images_path = os.listdir(test_path + '/images')

    # sort the list of images
    sortedimages = unet_preprocessing.sorted_alphanumeric(images_path)
    image_path = test_path + '/images/' + sortedimages[i]
    prediction = unet_segmentation.prediction(image_path, model)
    # plt.imshow(prediction)
    # plt.show()

    # read the mask
    mask = cv2.imread(test_path + '/masks/' + sortedimages[i], cv2.IMREAD_GRAYSCALE)
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    side = len(mask[0])

    # reshape the mask as a 1D array
    y_true_f = mask.reshape(side * side)
    y_pred_f = prediction.reshape(side * side)

    # compute the intersection and the union of true pixels
    intersection = sum(y_true_f * y_pred_f)
    sum_of_true = sum(y_true_f)
    sum_of_true += sum(y_pred_f)
    return intersection, sum_of_true


def test_model_3d(datasetpath, model_path):
    """ Test the model on the test set
    Args:
        datasetpath: path to the test set
        model: path to the model
    Return:
        test_acc: the average score of the model on the test set
    """

    test_path = datasetpath + '/test'
    masks_path = test_path + '/masks'

    # load the model
    model = tf.keras.models.load_model(model_path, compile=False)
    patient_intersection = 0
    patient_sumtrue = 0
    last_patient = 0
    dices = []
    beginning = True
    listmasks = os.listdir(masks_path)

    # sort the list of masks
    sortedmasks = unet_preprocessing.sorted_alphanumeric(listmasks)
    len_data = len(listmasks)

    # for each mask
    for i in range(len_data):

        # get the index of the patient
        patient_nbr = int(sortedmasks[i].split('_')[0])

        # if this is the first iteration
        if beginning:
            beginning = False
            # count intersection and sum of true for the mask and the prediction
            intersection, sum_of_true = count_prediction(datasetpath, model, i)
            # set this values to the patient intersection and sum of true
            patient_intersection = intersection
            patient_sumtrue = sum_of_true
            # set the index of the patient as the last patient that was counted
            last_patient = patient_nbr

        # else if the patient is the same as the last patient
        # and this is not the end of the dataset
        elif (patient_nbr == last_patient and i < len_data-1):
            # add the counts of the mask to the count of the current patient
            intersection, sum_of_true = count_prediction(datasetpath, model, i)
            patient_intersection += intersection
            patient_sumtrue += sum_of_true

        # else if this is the last element of the dataset
        elif i == len_data-1:
            # add the counts of the mask to the count of the current patient
            intersection, sum_of_true = count_prediction(datasetpath, model, i)
            patient_intersection += intersection
            patient_sumtrue += sum_of_true
            # compute and append the dice score of the patient
            dice = (2 * patient_intersection) / (patient_sumtrue)
            dices.append(dice)

        # else, if the patient is different from the last patient
        else:
            # compute and append the dice score of the last patient
            dice = (2 * patient_intersection) / (patient_sumtrue)
            dices.append(dice)
            # restart the counts for the new patient
            intersection, sum_of_true = count_prediction(datasetpath, model, i)
            patient_intersection = intersection
            patient_sumtrue = sum_of_true
            last_patient = patient_nbr

    print(dices)
    test_acc = np.mean(dices)
    # return the average of the dice scores
    return test_acc


def dice_3d(datasetpath, model, patient):
    test_path = datasetpath + '/test'
    masks_path = test_path + '/masks'
    model = tf.keras.models.load_model(model, compile=False)
    patient_intersection = 0.
    patient_sumtrue = 0.
    beginning = True
    listmasks = os.listdir(masks_path)
    sortedmasks = unet_preprocessing.sorted_alphanumeric(listmasks)
    len_data = len(listmasks)
    for i in range(len_data):
        patient_nbr = int(sortedmasks[i].split('_')[0])
        if patient_nbr >= patient:
            if beginning:
                beginning = False
                intersection, sum_of_true = count_prediction(datasetpath, model, i)
                patient_intersection = intersection
                patient_sumtrue = sum_of_true
            elif patient_nbr == patient:
                intersection, sum_of_true = count_prediction(datasetpath, model, i)
                patient_intersection += intersection
                patient_sumtrue += sum_of_true
            else:
                break
    dice = (2 * patient_intersection) / (patient_sumtrue)
    print('dice 3d: ' + str(dice))
    return dice


def adjustData(img, mask, class_train):
    """ adjust the data generated by the generator
    Args:
        img: list of images generated
        mask: list of maskq generated
        class_train: string equals to "train" if the data is from the train set
    Return:
        img: list of images adjusted
        mask: list of masks adjusted
    """

    # for each image
    for image in img:
        # Set the top left pixel to 0, because this pixel was sett to a small
        # value for the standardization of the range of pixels of the data
        image[0][0] = 0
        # print(np.max(image))
        # plt.imshow(image, cmap='gray', vmin=0, vmax=65535)
        # plt.show()

    # if the data is from the train set
    if class_train == 'train':
        # create slight brightness modification of the images
        brightness = random.uniform(0.985, 1.015) # tocheck
        for i in range(len(img)):
            img[i] = img[i] * brightness

    # whitening: normalize the data
    img = img-MEAN
    img = img/STD

    # range the pixels intensities between 0 and 1
    mask = mask / 255

    # plt.imshow(img[0], cmap='gray', interpolation='none')
    # plt.show()
    # plt.imshow(mask[0], cmap='gray', interpolation='none')
    # plt.show()

    # make sure that any value of the mask is 0 or 1
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    # return the normalized images and the masks
    return (img,mask)


def dataAugmentation(train_data_dir, class_train = 'train'):
    ''' Create a generator for the data augmentation.
        The generator is composed of two independent generators:
            - one for the images
            - one for the masks
        Both generators apply the same augmentation, to preserve the
        correctness of the mask with respect to the image.
    Args:
        train_data_dir: string, path to the directory of the data
        class_train: string, "train", "validation" or "test" to represent
                     if the data comes from the train set, validation set
                     or test set
    Return:
        train_generator: generator of the images and the masks
    '''

    # if the data is from the train set
    if(class_train == 'train'):

        # create the generators with the random transformations defined
        image_datagen = ImageDataGenerator(dtype=tf.uint16, zoom_range=0.08, rotation_range=25) # tocheck
        mask_datagen = ImageDataGenerator(dtype=tf.uint16, zoom_range=0.08, rotation_range=25)

        # define the options of the generators
        image_generator = image_datagen.flow_from_directory(
            train_data_dir + '/' + class_train,
            classes=['images'],
            class_mode=None,
            color_mode='grayscale',
            target_size=(512, 512),
            batch_size=1, #tocheck
            shuffle=True,
            seed=1)
        mask_generator = mask_datagen.flow_from_directory(
            train_data_dir + '/' + class_train,
            classes=['masks'],
            class_mode=None,
            color_mode='grayscale',
            target_size=(512, 512),
            batch_size=1, #tocheck
            shuffle=True,
            seed=1)

    # if the data is from the validation or the test set
    elif class_train == 'validation' or class_train == 'test':

        # create the generators without random transformations
        image_datagen = ImageDataGenerator(dtype=tf.uint16)
        mask_datagen = ImageDataGenerator(dtype=tf.uint16)

        # define the options of the generators
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

    # link the two generators into one object
    train_generator = zip(image_generator, mask_generator)

    # for every list of images and masks created
    for (img, mask) in train_generator:
        # adjust the data to the right format
        img, mask = adjustData(img, mask, class_train)
        yield (img, mask)

    # return the generator
    return train_generator


def simpleSGD(datasetpath, preloaded, epochs, name):
    ''' Simple SGD algorithm for 512x512 images with centralized data
    Args:
        datasetpath(string): path to the directory of the data
        preloaded(string): path to the preloaded weights, None if not preloaded
        epochs(int): max number of epochs
        name(string): name of the model to save
    Return:
        model: model fitted
    '''

    model_name = name + '.h5'

    # create a checkpointer to save the model only if the score is the best over each epoch
    checkpointer = ModelCheckpoint(model_name, verbose=1, save_best_only=True)

    # create history object to save the history of the training
    history = History()

    # use the checkpointer to create callback, making the algorithm stop if the score stops
    # improving for more than 10 epochs
    callbacks = [
        EarlyStopping(patience=10, monitor='val_loss', mode=min), #tocheck
        TensorBoard(log_dir='logs'),
        history,
        checkpointer,
    ]

    # create Adam optimizer
    optimizer = tf.keras.optimizers.Adam

    # define the loss function
    loss_metric = dice_coef_loss # tocheck

    # define the accuracy metric
    metrics = [dice_coef] #tocheck

    # lr = lr_scheduler.TanhDecayScheduler()

    # define the learning rate
    lr = 1e-5 #tocheck

    # create the U-net model
    model = get_model()

    # load the weights if a model was passed as argument
    if preloaded is not None:
        model.load_weights(preloaded)

    # compile the model
    model.compile(optimizer=optimizer(learning_rate=lr), loss=loss_metric, metrics=metrics)

    model.summary()

    # create the generators for training and validation
    training_generator = dataAugmentation(datasetpath, class_train='train')
    validation_generator = dataAugmentation(datasetpath, class_train='validation')

    # store the number of elements in the training Sand validation sets
    len_training = len(os.listdir(datasetpath + '/train/images'))
    len_validation = len(os.listdir(datasetpath + '/validation/images'))

    # fit the model
    hist = model.fit(training_generator, validation_data=validation_generator,
                     validation_steps=len_validation/1, steps_per_epoch=len_training/1,
                     epochs=epochs, shuffle=True, callbacks=callbacks, verbose=1)       # tocheck len attention

    print((hist.history['val_dice_coef']))

    measures_file = name + "_data.txt"

    # save the measures of the training in a file
    train_accs = hist.history['dice_coef']
    val_accs = hist.history['val_dice_coef']
    file_utils.write_measures(measures_file, train_accs)
    file_utils.write_measures(measures_file, val_accs)

    # plot the graph of the training and validation accuracy
    plots.history(train_accs, val_accs, name)

    # return the model
    return model


def get_ratio_of_clients(nbrclients, datasetpath):
    """ Calculates the proportion of a client’s local training data with the overall training data held by all clients
    Args:
        nbrclients: number of clients
        datasetpath: path to the dataset with all clients
    Return:
        ratio: array with the ratio of training data for each client
    """

    sizes = []
    totalsize = 0

    # for every client
    for i in range(nbrclients):
        client_path = datasetpath + '/' + str(i) + "/train/images"
        # get the size of the client's local training data
        client_size = len(os.listdir(client_path))
        # append its size to an array
        sizes.append(client_size)
        # add its size to the total size
        totalsize += client_size

    sizes = np.array(sizes)
    # ponderate the size of each client with the total size
    ratios = sizes / totalsize
    return ratios


def scale_model_weights(weights, scalar):
    """ Scales an array of weights
    Args:
        weight: array of weights
        scalar: scalar to multiply the array
    Return:
        scaled_weight: scaled array of weights
    """
    weight_final = []
    steps = len(weights)
    # for each weight
    for i in range(steps):
        # append the scaled weight to the final array
        weight_final.append(scalar * weights[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    """ Return the sum of the listed scaled weights. This is equivalent to scaled avg of the weights
    Args:
        scaled_weight_list: list of scaled weights
    Return:
        avg_grad: sum of the scaled weights
    """

    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    return avg_grad



def fedEq(datasetpath, preloaded, nbrclients, name, frac = 1, epo = 1, comms_round = 100, patience = 10): # tocheck
    """ federated equal-chances algotihm
        Args:
            datasetpath: path to the dataset
            preloaded: path to the preloaded model, None if no model is preloaded
            nbrclients: number of clients
            name: name of the model
            frac: fraction of the number of clients to use at each round
            epo: number of epochs at each round
            comms_round: number of communication rounds
            patience: number of rounds to wait before stopping the training
        Return:
            global_model: the model trained using federated equal-chances
    """

    model_name = name + '.h5'

    # store values useful for the training
    best_val_acc = 0.
    val_accs = []
    train_acc = []
    global_val_accs = []
    patience_wait = 0

    # create an array of the ratios of the training data of each client
    clients_weight = get_ratio_of_clients(nbrclients, datasetpath)
    biggest_client = np.argmax(clients_weight)

    print("clients_weight: " + str(clients_weight))

    # process and batch the training data for each client
    # clients_batched = dict()
    # for (client_name, data) in clients.items():
    #     clients_batched[client_name] = batch_data(data, bs = bs)

    # validation_generator = dataAugmentation(datasetpath, class_train='validation')

    # create history object to store the training history
    history = History()

    # create the callback using the history object
    callbacks = [
        history,
    ]

    # create the Adam optimizer
    optimizer = tf.keras.optimizers.Adam

    # define the loss function
    loss_metric = dice_coef_loss

    # define the accuracy metric
    metrics = [dice_coef]

    # define the learning rate
    lr = 1e-5 # tocheck

    # initialize U-Net global model
    global_model = get_model()

    # if a preloaded model is provided, load its weights
    if preloaded is not None:
        global_model.load_weights(preloaded)

    # compile the global model
    global_model.compile(optimizer=optimizer(learning_rate=lr), loss=loss_metric, metrics=metrics)

    # start global training loop
    for comm_round in range(comms_round):

        print('comm_round: ' + str(comm_round))

        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()

        # initial list to collect local model weights after scaling
        scaled_local_weight_list = list()

        # list the index of all clients
        clients = list(range(0, nbrclients))

        # if frac is not equal to 1, the clients must be shuffled
        # random.shuffle(clients)
        print('clients: ' + str(clients))

        # compute the number of clients to use at each round
        nbrclients = int(frac * nbrclients)

        # for each client used in the round
        for client in clients[:nbrclients]:

            # get the directory of the client
            client_path = datasetpath + '/' + str(client)

            # create a new local model for the client
            local_model = get_model()

            # compile the local model
            local_model.compile(optimizer=optimizer(learning_rate=lr), loss=loss_metric, metrics=metrics)

            # set local model weight to the weight of the global model
            local_model.set_weights(global_weights)

            # create a new data generator for the client
            training_generator = dataAugmentation(client_path, class_train='train')

            # get the length of the training data of the client
            len_training = len(os.listdir(client_path + '/train/images'))

            # fit the local model on the client data. The number of images used for training is
            # equals to the size of clients_weight[1] as it is the biggest client in this case
            train_acc = local_model.fit(training_generator,
                                        steps_per_epoch=len_training*(clients_weight[biggest_client]/clients_weight[client]),
                                        epochs=epo, shuffle=True,
                                        callbacks=callbacks, verbose=1)

            # scaling factor equals 1/nbrclients because all clients will generate the same number of images
            scaling_factor = 1./nbrclients

            # scale the local model weights and add it to the list of scaled weights
            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)

            # clear session to free memory after each communication round
            K.clear_session()

        # to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = sum_scaled_weights(scaled_local_weight_list)

        # update global model
        global_model.set_weights(average_weights)

        mean_val_acc = 0.
        # for each client used in the round
        for client in clients[:nbrclients]:

            # create the path to the client
            client_path = datasetpath + '/' + str(client)

            # create a validation data generator for the client and get the number of images in the validation set
            validation_generator = dataAugmentation(client_path, class_train='validation')
            len_validation = len(os.listdir(client_path + '/validation/images'))

            # evaluate the new global model on the client validation data
            val_acc = global_model.evaluate_generator(generator=validation_generator, steps=len_validation/1)

            # get the Dice coefficient and appends it in the list of validation accuracies
            val_acc=val_acc[1] # Dice's coeff
            mean_val_acc+=val_acc
            val_accs.append(val_acc)

        # compute the mean validation accuracy over all the clients
        mean_val_acc = mean_val_acc / nbrclients

        # if the mean validation accuracy is better than the best one
        if mean_val_acc > best_val_acc:
            print("Average validation Dice score improved: from " + str(best_val_acc) + " to: " + str(mean_val_acc))
            # update the best validation accuracy
            best_val_acc = mean_val_acc
            # restart patience
            patience_wait = 0
            # save the model
            global_model.save(model_name)
        # else increment the patience
        else:
            print("Average validation Dice score didn't improve, got a dice score of " + str(mean_val_acc) + " but best is still " + str(best_val_acc) )
            patience_wait += 1
        print("patience_wait = " + str(patience_wait))

        # Stores global_validation_acc
        # global_val_path = "datasets/dataset_lung" # tocheck
        # validation_generator = dataAugmentation(global_val_path, class_train='validation')
        # len_validation = len(os.listdir(global_val_path + '/validation/images'))
        # global_val_acc = global_model.evaluate_generator(generator=validation_generator, steps=len_validation / 1)
        # global_acc = global_val_acc[1]
        # global_val_accs.append(global_acc)

        # if the patience is reached, stop the training
        if patience_wait >= patience:
            break

        print('Accuracy after {} rounds = {}'.format(comm_round, mean_val_acc))

    print("FINAL ACCURACY: " + str(mean_val_acc))

    # get the training accuracy
    train_accs = train_acc.history['dice_coef']

    # create a file to store the training details
    measures_file = name + "_data.txt"

    # write the training details in the file
    file_utils.write_measures(measures_file, train_accs)
    file_utils.write_measures(measures_file, val_accs)
    # file_utils.write_measures(measures_file, global_val_accs)

    # plot the graph of the training and validation accuracies
    plots.history_fedavg(train_accs, val_accs, len(clients), name)

    # return the global model trained
    return global_model


def fedAvg(datasetpath, preloaded, nbrclients, name, frac = 1, epo = 1, comms_round = 100, patience = 10): # tocheck
    """ federated averaging algorithm
        Args:
            datasetpath: path to the dataset
            preloaded: path to the preloaded model, None if no model is preloaded
            nbrclients: number of clients
            name: name of the model
            frac: fraction of the clients to use at each round
            epo: number of epochs at each round
            comms_round: number of rounds
            patience: patience to wait before stopping the training
        Return:
            global_model: the global model trained
    """

    model_name = name + '.h5'

    # create values useful for the training
    best_val_acc = 0.
    val_accs = []
    train_acc = []
    global_val_accs = []
    patience_wait = 0

    # array with the ratio of size of the clients
    clients_weight = get_ratio_of_clients(nbrclients, datasetpath)

    print("clients_weight: " + str(clients_weight))

    # process and batch the training data for each client
    # clients_batched = dict()
    # for (client_name, data) in clients.items():
    #     clients_batched[client_name] = batch_data(data, bs = bs)

    # validation_generator = dataAugmentation(datasetpath, class_train='validation')

    # create history object to store the training details
    history = History()

    # create the callbacks with the history object
    callbacks = [
        history,
    ]

    # create the Adam optimizer
    optimizer = tf.keras.optimizers.Adam

    # define the loss function
    loss_metric = dice_coef_loss

    # define the accuracy metric
    metrics = [dice_coef]

    # define the learning rate
    lr = 1e-5 # tocheck

    # initialize global model
    global_model = get_model()

    # if a preloaded model is given, load it
    if preloaded is not None:
        global_model.load_weights(preloaded)

    # compile the global model
    global_model.compile(optimizer=optimizer(learning_rate=lr), loss=loss_metric, metrics=metrics)

    # start global training loop
    for comm_round in range(comms_round):

        print('comm_round: ' + str(comm_round))

        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()

        # initial list to collect local model weights after scaling
        scaled_local_weight_list = list()

        # create the list with the indexes of the clients
        clients = list(range(0, nbrclients))

        # if the fraction is not 1, shuffle the clients
        # random.shuffle(clients)
        print('clients: ' + str(clients))

        # get the number of clients to use in the round
        nbrclients = int(frac * nbrclients)

        # for each client used in the round
        for client in clients[:nbrclients]:

            # get the path to the client's data
            client_path = datasetpath + '/' + str(client)

            # create the local U-Net model of the client
            local_model = get_model()

            # compile the local model of the client
            local_model.compile(optimizer=optimizer(learning_rate=lr), loss=loss_metric, metrics=metrics)

            # set local model weights to the weights of the global model
            local_model.set_weights(global_weights)

            # create a generator for the client's data
            training_generator = dataAugmentation(client_path, class_train='train')

            # get the length of the training data
            len_training = len(os.listdir(client_path + '/train/images'))

            # fit the local model on the client's data, with a number of images
            # generated equal to the length of the training data of the client
            train_acc = local_model.fit(training_generator,
                                       steps_per_epoch=len_training, epochs=epo, shuffle=True,
                                       callbacks=callbacks, verbose=1)

            # get the scaling factor of the client
            scaling_factor = clients_weight[client]

            # scale the weights with the size of the client
            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)

            # clear session to free memory after each communication round
            K.clear_session()

        # to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = sum_scaled_weights(scaled_local_weight_list)

        # update global model
        global_model.set_weights(average_weights)

        mean_val_acc = 0.

        # for each client used in the round
        for client in clients[:nbrclients]:
            # get the path to the client's data
            client_path = datasetpath + '/' + str(client)

            # create a validation generator for the client's validation data
            validation_generator = dataAugmentation(client_path, class_train='validation')

            # get the length of the validation data
            len_validation = len(os.listdir(client_path + '/validation/images'))

            # evaluate the score of the global model on the client's validation data
            val_acc = global_model.evaluate_generator(generator=validation_generator, steps=len_validation/1)

            # get the Dice coefficient of the global model on the client's validation data
            val_acc=val_acc[1] # Dice's coeff

            # save the validation accuracy of the global model on the client
            mean_val_acc+=val_acc
            val_accs.append(val_acc)

        # save the mean validation accuracy of the global model over all clients
        mean_val_acc = mean_val_acc / nbrclients

        # if best acc so far
        if mean_val_acc > best_val_acc:
            print("Average validation Dice score improved: from " + str(best_val_acc) + " to: " + str(mean_val_acc))
            # update the best validation accuracy
            best_val_acc = mean_val_acc
            # reset patience
            patience_wait = 0
            # save the global model
            global_model.save(model_name)
        # else, increment patience
        else:
            print("Average validation Dice score didn't improve, got a dice score of " + str(mean_val_acc) + " but best is still " + str(best_val_acc) )
            patience_wait += 1
        print("patience_wait = " + str(patience_wait))

        # Stores global_validation_acc
        # global_val_path = "datasets/dataset_lung" # tocheck
        # validation_generator = dataAugmentation(global_val_path, class_train='validation')
        # len_validation = len(os.listdir(global_val_path + '/validation/images'))
        # global_val_acc = global_model.evaluate_generator(generator=validation_generator, steps=len_validation / 1)
        # global_acc = global_val_acc[1]
        # global_val_accs.append(global_acc)

        # if patience limit is reached, stop the process
        if patience_wait >= patience:
            break

        print('Accuracy after {} rounds = {}'.format(comm_round, mean_val_acc))

    print("FINAL ACCURACY: " + str(mean_val_acc))

    # get the history of the global model's accuracy
    train_accs = train_acc.history['dice_coef']

    # create a file to store the measures
    measures_file = name + "_data.txt"

    # write the measures in the file
    file_utils.write_measures(measures_file, train_accs)
    file_utils.write_measures(measures_file, val_accs)
    # file_utils.write_measures(measures_file, global_val_accs)

    # plot the history of the global model's accuracy
    plots.history_fedavg(train_accs, val_accs, len(clients), name)

    return global_model

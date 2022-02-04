import gc
import pickle
import numpy as np
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import History
from tensorflow.keras import backend as K
import tensorflow as tf
import dcm_contour
import os
import seaborn
import matplotlib.pyplot as plt
import cv2
import random
import plots
import imageio
import lr_scheduler



MEAN = 4507.478941282831
STD = 7182.589254997573


def get_mean_std(dataset_path):
    means = []
    stds = []
    images = os.listdir(dataset_path)
    for name in images:
        image_file = dataset_path + '/' + name
        image = imageio.imread(image_file)
        image[0][0] = 0
        mean = np.mean(image)
        std = np.std(image)
        means.append(mean)
        stds.append(std)
        print('mean: ' + str(mean))
        print('std: ' + str(std))
    return np.mean(means), np.mean(std)


smooth = 1.
# Dice Coefficient to work with Tensorflow
def dice_coef_ponderated(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)) * K.sum(y_true)


def dice_coef_loss_ponderated(y_true, y_pred):
    return -dice_coef_ponderated(y_true, y_pred)


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



def test_model(x_test, y_test, model):
    score = 0
    nbrelems = 0
    test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))
    for (X_test, Y_test) in test_batched:
        predictions = model.predict(X_test, batch_size=3)
        Y_test = np.array(Y_test)
        for i in range(len(Y_test)):
            coef = dice_coef_2(Y_test[i], predictions[i])
            score += coef
            nbrelems += 1
        K.clear_session()
    return score / nbrelems


def test_model_ponderated(x_test, y_test, model):
    score = 0
    nbrpixels = 0
    test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))
    for (X_test, Y_test) in test_batched:
        predictions = model.predict(X_test, batch_size=3)
        Y_test = np.array(Y_test)
        for i in range(len(Y_test)):
            coef = dice_coef_2(Y_test[i], predictions[i])
            score += coef * np.sum(Y_test[i])
            nbrpixels += np.sum(Y_test[i])
        K.clear_session()
    return score / nbrpixels


def adjustData(img, mask):
    img[0][0] = 0
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


def dataAugmentation(train_data_dir, class_train = 'train', batch_size = 3):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''

    if(class_train == 'train'):
        image_datagen = ImageDataGenerator(zoom_range=0.05, rotation_range=5)
        mask_datagen = ImageDataGenerator(zoom_range=0.05, rotation_range=5)
    elif(class_train == 'validation'):
        image_datagen = ImageDataGenerator()
        mask_datagen = ImageDataGenerator()

    image_generator = image_datagen.flow_from_directory(
        train_data_dir + '/' + class_train,
        classes=['images'],
        class_mode=None,
        color_mode='grayscale',
        target_size=(512, 512),
        batch_size=batch_size,
        seed=1)
    mask_generator = mask_datagen.flow_from_directory(
        train_data_dir + '/' + class_train,
        classes=['masks'],
        class_mode=None,
        color_mode='grayscale',
        target_size=(512, 512),
        batch_size=batch_size,
        seed=1)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img,mask)
        yield (img, mask)

    return train_generator






def simpleSGD(datasetpath, epochs):
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


    checkpointer = ModelCheckpoint('best_val_loss.h5', verbose=1, save_best_only=True)

    history = History()

    callbacks = [
        EarlyStopping(patience=15, monitor='val_loss', mode=min),
        TensorBoard(log_dir='logs'),
        history,
        checkpointer,
    ]

    optimizer = tf.keras.optimizers.Adam
    loss_metric = dice_coef_loss
    metrics = [dice_coef, dice_coef_ponderated, 'accuracy']
    lr = lr_scheduler.TanhDecayScheduler()
    batch_size = 3

    model = get_model()

    model.compile(optimizer=optimizer(learning_rate=lr), loss=loss_metric, metrics=metrics) # Check

    training_generator = dataAugmentation(datasetpath, class_train='train', batch_size=batch_size)
    validation_generator = dataAugmentation(datasetpath, class_train='validation', batch_size=batch_size)

    len_training = len(os.listdir(datasetpath + 'train/images'))
    len_validation = len(os.listdir(datasetpath + 'validation/images'))

    hist = model.fit_generator(training_generator, validation_data=validation_generator, validation_steps=len_validation/batch_size, steps_per_epoch=len_training/batch_size, epochs=epochs, shuffle=True, callbacks=callbacks)

    # Train the model, doing validation at the end of each epoch.
    # hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), shuffle=True, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    print((hist.history['val_dice_coef']))

    # test_model(x_train, y_train, model)

    plots.history(hist)
    return model



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
        x_train_client, y_train_client, x_test_client, y_test_client = prepareTrainTest(datasetpath)
        clientname = "client_" + str(clientnbr)
        data = list(zip(x_train_client, y_train_client))
        clients[clientname] = data
        x_test.extend(x_test_client)
        y_test.extend(y_test_client)
        clientnbr+=1
    return clients, x_test, y_test



def batch_data(data_shard, bs=3):
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


def get_ratio_of_clients(nbrclients, datasetpath):
    ''' Calculates the proportion of a client’s local training data with the overall training data held by all clients
        args:
            nbrclients: number of clients
            datasetpath: path to the dataset with all clients
    '''
    sizes = []
    totalsize = 0

    for i in range(nbrclients):
        client_path = datasetpath + str(i) + "/train/images"
        client_size = len(os.listdir(client_path))
        sizes.append(client_size)
        totalsize += client_size

    sizes = np.array(sizes)
    return sizes/totalsize



def weight_scaling_factor(clients_trn_data, client_name, frac):
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



def fedAvg(datasetpath, nbrclients, frac = 1, bs = 3, epo = 1, comms_round = 200, patience = 15):
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

    best_test_acc = 0
    test_accs = []
    train_acc = []
    patience_wait = 0
    len_validation = len(os.listdir(datasetpath + 'validation/images'))

    clients_weight = get_ratio_of_clients(nbrclients, datasetpath)
    print("clients_weight: " + str(clients_weight))

    # process and batch the training data for each client
    # clients_batched = dict()
    # for (client_name, data) in clients.items():
    #     clients_batched[client_name] = batch_data(data, bs = bs)

    validation_generator = dataAugmentation(datasetpath, class_train='validation', batch_size=bs)

    history = History()

    callbacks = [
        history,
    ]

    optimizer = tf.keras.optimizers.Adam
    loss_metric = dice_coef_loss
    metrics = [dice_coef, "accuracy"]
    lr = 1e-4

    # initialize global model
    global_model = get_model()

    global_model.compile(optimizer=optimizer(learning_rate=lr), loss=loss_metric, metrics=metrics)  # Check


    # commence global training loop
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

        random.shuffle(clients)
        print('clients: ' + str(clients))

        # loop through each client and create new local model
        nbrclients = int(frac * nbrclients)
        # for client in client_names[:nbrclients]:
        for client in clients[:nbrclients]:

            client_path = datasetpath + str(client)

            local_model = get_model()

            local_model.compile(optimizer=optimizer(learning_rate=lr), loss=loss_metric, metrics=metrics)

            # set local model weight to the weight of the global model
            local_model.set_weights(global_weights)

            training_generator = dataAugmentation(client_path, class_train='train', batch_size=bs)

            len_training = len(os.listdir(client_path + '/train/images'))

            train_acc = local_model.fit_generator(training_generator,
                                       steps_per_epoch=len_training / bs, epochs=epo, shuffle=True,
                                       callbacks=callbacks)

            # scale the model weights and add to list

            # scaling_factor = weight_scaling_factor(clients_batched, client, frac)

            scaling_factor = clients_weight[client]

            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)

            # clear session to free memory after each communication round
            K.clear_session()
            gc.collect()

        # to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = sum_scaled_weights(scaled_local_weight_list)

        # update global model
        global_model.set_weights(average_weights)

        # process and batch the test se
        # test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

        # test global model and print out metrics after each communications round
        # for (X_test, Y_test) in test_batched:

        # test_acc = test_model(X_test, Y_test, global_model)
        test_acc = global_model.evaluate_generator(generator=validation_generator, steps=len_validation/bs)
        print(type(test_acc))
        print('test_acc: ' + str(test_acc))
        print(len(test_acc))

        test_acc=test_acc[1] # Dice's coeff
        print(test_acc)

        test_accs.append(test_acc)

        # If best acc so far
        if test_acc > best_test_acc:
            print("Dice score improved: from " + str(best_test_acc) + " to: " + str(test_acc))
            best_test_acc = test_acc
            patience_wait = 0
            global_model.save('fedAvg_best_model.h5')
        else:
            print("Dice score didn't improve, got a dice score of " + str(test_acc) + " but best is still " + str(best_test_acc) )
            patience_wait += 1
        print("patience_wait = " + str(patience_wait))


        if patience_wait >= patience:
            break

        print('Accuracy after {} rounds = {}'.format(comm_round, test_acc))

    print("FINAL ACCURACY: " + str(test_acc))
    print(train_acc.history['dice_coef'])
    print(test_accs)

    plots.history_fedavg(train_acc.history['dice_coef'], test_accs, len(clients))

    return global_model



def show_rtstruct(organ, dcm_path, img_nbr):
    if organ == "tumor":
        # plot the rt struct of the image
        index = dcm_contour.get_index(dcm_path, "GTV-1")
        images, contours, _ = dcm_contour.get_data(dcm_path, index=index)
        for img_arr, contour_arr in zip(images[img_nbr:img_nbr+1], contours[img_nbr:img_nbr+1]):
            dcm_contour.plot2dcontour(img_arr, contour_arr, img_nbr)
        contours = np.array(contours)
        cntr = contours[img_nbr, 0]
    if organ == "esophagus":
        # plot the rt struct of the image
        index = dcm_contour.get_index(dcm_path, "Esophagus")
        images, contours, _ = dcm_contour.get_data(dcm_path, index=index)
        for img_arr, contour_arr in zip(images[img_nbr:img_nbr+1], contours[img_nbr:img_nbr+1]):
            dcm_contour.plot2dcontour(img_arr, contour_arr, img_nbr)
        contours = np.array(contours)
        cntr = contours[img_nbr, 0]
    elif organ == "heart":
        index = dcm_contour.get_index(dcm_path, "Heart")
        images, contours, _ = dcm_contour.get_data(dcm_path, index=index)
        for img_arr, contour_arr in zip(images[img_nbr:img_nbr+1], contours[img_nbr:img_nbr+1]):
            dcm_contour.plot2dcontour(img_arr, contour_arr, img_nbr)
        cntr = contours[img_nbr][0]
        for i in range(1, len(contours[img_nbr])):
            cntr += contours[img_nbr][i]
    elif organ == "lung":
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
            cntr += contours1[img_nbr][i]
    plt.show()
    return cntr



def heatMap(predictions):
    """ Plot the heat map of the tumor predictions"""
    fig, ax = plt.subplots()
    # title = "Prediction heatmap"
    # plt.title(title, fontsize=18)
    # ttl = ax.title
    # ttl.set_position([0.5, 1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    seaborn.heatmap(predictions, ax=ax)
    plt.show()




def finalPrediction(cntr, predictions):
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            if predictions[(i, j)] >= 0.5 and cntr[(i, j)] != 1:
                cntr[(i, j)] = 10.0
            elif cntr[(i, j)] == 1:
                cntr[(i, j)] = 50.0
    # title = zone + " prediction"
    # plt.title(title, fontsize=18)
    # plt.imshow(cntr, cmap='gray')
    # plt.show()
    # plt.imshow(cntr, cmap='plasma')
    # plt.show()
    # plt.imshow(cntr, cmap='viridis')
    # plt.show()
    # plt.imshow(cntr, cmap='seismic')
    # plt.show()
    plt.imshow(cntr, cmap='magma')
    plt.show()



def segmentation_2d(model, client_path, mask_path, image_path, img_nbr, organ):
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

    cntr = show_rtstruct(organ, dcm_path, img_nbr)

    image = imageio.imread(image_path)
    image[0][0] = 0
    image = image-MEAN
    image = image/STD
    image = np.reshape(image, (1, 512, 512, 1))
    predictions = model.predict(image)
    predictions = np.reshape(predictions, (512, 512))

    heatMap(predictions)

    finalPrediction(cntr, predictions)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask[mask < 0.5] = 0  # Set out of tumor to 0
    mask[mask > 0.5] = 1  # Set out of tumor to 1
    dice = dice_coef_2(mask, predictions)
    print("dice accuracy: " + str(dice))

    return cntr, predictions
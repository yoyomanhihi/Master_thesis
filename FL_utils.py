
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
import copy

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

def load(paths, verbose=-1):
    '''expects images for each class in seperate dir,
    e.g all digits in 0 class in the directory named 0 '''
    data = list()
    labels = list()
    # loop over the input images
    for (i, imgpath) in enumerate(paths):
        # load the image and extract the class labels
        im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        image = np.array(im_gray)#.flatten() #CHANGED
        label = imgpath.split(os.path.sep)[-2]
        # scale the image to [0, 1] and add to list
        data.append(image / 255)
        labels.append(label)
        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))
    # return a tuple of the data and labels
    return data, labels


def prepareTrainTest(path):
    ''' return:
            X_train: training set data
            X_test: test set data
            y_train: training set labels
            y_test: test set labels
        args:
            path: filepath to the data
    '''

    # declear path to your mnist data folder
    img_path = path

    # get the path list using the path object
    image_paths = list(paths.list_images(img_path))

    # apply our function
    image_list, label_list = load(image_paths, verbose=10000)
    # print(image_list[0])
    # print(label_list)

    # binarize the labels
    lb = LabelBinarizer()
    label_list = lb.fit_transform(label_list)
    # print(label_list)

    image_list = np.reshape(image_list, (len(image_list), 28, 28, 1))

    # split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(image_list,
                                                        label_list,
                                                        test_size=0.2,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def prepareAllData(path):
    ''' return:
            X_train: training set data
            X_test: test set data
            y_train: training set labels
            y_test: test set labels
        args:
            path: filepath to the data
    '''

    # declear path to your mnist data folder
    img_path = path

    # get the path list using the path object
    image_paths = list(paths.list_images(img_path))

    # apply our function
    image_list, label_list = load(image_paths, verbose=10000)
    image_list = np.array(image_list)

    image_list = np.reshape(image_list, (len(image_list), 28, 28, 1)) # Only for CNN

    # binarize the labels
    lb = LabelBinarizer()
    label_list = lb.fit_transform(label_list)
    # print(label_list)

    # Shuffle data and labels the same way
    p = np.random.permutation(len(label_list))
    return image_list[p], label_list[p]



def splitByLabel(data, labels):
    dico = {}
    for i in range(len(labels)):
        if labels[i] not in dico.keys():
            dico[labels[i]] = [data[i]]
        else:
            dico[labels[i]].append(data[i])
    for key in dico.keys():
        random.shuffle(dico[key])
    return dico



def create_clients(image_list, label_list, num_clients=10, initial='client'):
    ''' return: a dictionary with keys clients' names and value as
                data shards - tuple of images and label lists.
        args:
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of federated members (clients)
            initials: the clients'name prefix, e.g, clients_1

    '''

    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]

    # randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    # shard data and place at each client
    size = len(data) // num_clients
    shards = [data[i:i + size] for i in range(0, size * num_clients, size)]

    # number of clients must equal number of shards
    assert (len(shards) == len(client_names))

    return {client_names[i]: shards[i] for i in range(len(client_names))}


def create_clients_non_iid(image_list, label_list, num_clients=100, initial='client', num_classes= 2):
    ''' return: a dictionary with keys clients' names and value as
                data shards - tuple of images and label lists.
        args:
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of federated members (clients)
            initials: the clients'name prefix, e.g, clients_1

    '''

    # Sort the data by labels
    data = list(zip(image_list, label_list))
    mergeSort(data)

    # shard data and randomize it
    size = len(data) // (num_clients*num_classes) # 37800 // 100*2 = 189
    shards = [data[i:i + size] for i in range(0, size * num_clients * num_classes, size)]
    random.shuffle(shards)

    # number of clients must equal number of shards
    assert (len(shards) == num_clients*num_classes)

    #Give shards to clients
    clients = {}
    for i in range(len(shards)):
        clientnbr = i // num_classes
        clientname = '{}_{}'.format(initial, clientnbr)
        if clientname not in clients:
            clients[clientname] = shards[i]
        else:
            clients[clientname].extend(shards[i])

    return clients


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


class SimpleMLP:
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
        model.add(Dense(classes, activation='softmax')),
        '''
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        '''
        return model


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


def test_model(X_test, Y_test, model, comm_round):
    """ Calculates the accuracy and the loss of the model
        args:
            X_test: test set data
            y_test: test set labels
            model: the model
            comm_round: number of total communication rounds
        return:
            acc: accuracy of the model
            loss: global loss of the model
    """
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss


def biggerLabel(label1, label2):
    ''' return: True if label1 is bigger than label 2
        args:
            label1 and label2: One hot encoding of labels
    '''
    for i in range(len(label1)):
        if label1[i] == 1 and label2[i] == 0:
            return True
        elif label2[i] == 1:
            return False


def mergeSort(arr):
    ''' Sort an array of one hot encoded labels '''
    if len(arr) > 1:
        # Finding the mid of the array
        mid = len(arr) // 2
        # Dividing the array elements
        L = arr[:mid]
        # into 2 halves
        R = arr[mid:]
        # Sorting the first half
        mergeSort(L)
        # Sorting the second half
        mergeSort(R)
        i = j = k = 0
        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if biggerLabel(L[i][1], R[j][1]):
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1



def simpleSGD(X_train, y_train, X_test, y_test, lr = 0.01, comms_round = 100):
    ''' Simple SGD algorithm
        args:
            clients: dictionary of the clients and their data
            X_test: test set data
            y_test: test set labels
        returns:
            SGD_acc: the global accuracy after comms_round rounds
    '''
    # # reshaping
    # # this assumes our data format
    # # For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while
    # # "channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
    # if k.image_data_format() == 'channels_first':
    #     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    #     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    #     input_shape = (1, img_rows, img_cols)
    # else:
    #     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    #     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    #     input_shape = (img_rows, img_cols, 1)
    # # more reshaping
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255
    # print('X_train shape:', X_train.shape)  # X_train shape: (60000, 28, 28, 1)





    test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))

    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    optimizer = SGD(lr=lr,
                    decay=lr / comms_round,
                    momentum=0.9
                    )

    SGD_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(320)
    print("dataset shape: " + str(np.shape(SGD_dataset)))
    smlp_SGD = SimpleMLP()
    # SGD_model = smlp_SGD.build(784, 10)
    SGD_model = smlp_SGD.build((28, 28, 1), 10)

    SGD_model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    # fit the SGD training data to model
    print(np.shape(SGD_dataset))
    _ = SGD_model.fit(SGD_dataset, epochs=100, verbose=2)

    #test the SGD global model and print out metrics
    for(X_test, Y_test) in test_batched:
            SGD_acc, SGD_loss = test_model(X_test, Y_test, SGD_model, 1)

    return SGD_acc


def fedAvg(clients, X_test, y_test, frac = 1, bs = 32, epo = 1, lr = 0.01, comms_round = 100):
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
    print(clients['client_1'])
    for (client_name, data) in clients.items():
        clients_batched[client_name] = batch_data(data, bs = bs)

    # process and batch the test set
    test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))


    loss='categorical_crossentropy'
    metrics = ['accuracy']
    optimizer = SGD(lr=lr,
                    decay=lr / comms_round,
                    momentum=0.9
                   )


    # initialize global model
    smlp_global = SimpleMLP()
    global_model = smlp_global.build((28, 28, 1), 10)

    # commence global training loop
    for comm_round in range(comms_round):

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
            local_model = smlp_local.build((28, 28, 1), 10)
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
            global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)

    return global_acc

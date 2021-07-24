
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
        image = np.array(im_gray).flatten()
        label = imgpath.split(os.path.sep)[-2]
        # scale the image to [0, 1] and add to list
        data.append(image / 255)
        labels.append(label)
        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))
    # return a tuple of the data and labels
    return data, labels


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

    # create a list of client names
    print(label_list)
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


def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
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
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model


def weight_scalling_factor(clients_trn_data, client_name):
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
    return local_count / global_count


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
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss


def non_iid_x(image_list, label_list, x=1, num_intraclass_clients=10):
    ''' creates x non_IID clients
    args:
        image_list: python list of images or data points
        label_list: python list of labels
        x: none IID severity, 1 means each client will only have one class of data
        num_intraclass_client: number of sub-client to be created from each none IID class,
        e.g for x=1, we could create 10 further clients by splitting each class into 10

    return - dictionary
        keys - clients's name,
        value - client's non iid 1 data shard (as tuple list of images and labels) '''

    non_iid_x_clients = dict()

    # create unique label list and shuffle
    unique_labels = np.unique(np.array(label_list))
    random.shuffle(unique_labels)

    # create sub label lists based on x
    sub_lab_list = [unique_labels[i:i + x] for i in range(0, len(unique_labels), x)]

    for item in sub_lab_list:
        class_data = [(image, label) for (image, label) in zip(image_list, label_list) if label in item]

        # decouple tuple list into seperate image and label lists
        images, labels = zip(*class_data)

        # create formated client initials
        initial = ''
        for lab in item:
            initial = initial + lab + '_'

        # create num_intraclass_clients clients from the class
        intraclass_clients = create_clients(list(images), list(labels), num_intraclass_clients, initial)

        # append intraclass clients to main clients'dict
        non_iid_x_clients.update(intraclass_clients)

    return non_iid_x_clients


def biggerLabel(label1, label2):
    for i in range(len(label1)):
        if label1[i] == 1 and label2[i] == 0:
            return True
        elif label2[i] == 1:
            return False


def mergeSort(arr):
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

    # Later for cross validation
    '''
    # Divide the data into 5 folds sorted by labels
    datalist = []
    size = len(data)
    for i in range(5):
        temp_data = (data[i:int(i+(size/5))])
        mergeSort(temp_data)
        datalist.append(temp_data)
    # print(datalist[4][5000][1])
    '''


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
        if clientnbr not in clients:
            clients[clientname] = shards[i]
        else:
            clients[clientname].extend(shards[i])

    return clients
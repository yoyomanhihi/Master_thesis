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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import local_FL_utils as FL_utils



#declear path to your mnist data folder
img_path = 'trainingSet/trainingSet'

#get the path list using the path object
image_paths = list(paths.list_images(img_path))

#apply our function
image_list, label_list = FL_utils.load(image_paths, verbose=10000)
# print(image_list[0])
# print(label_list)


#binarize the labels
lb = LabelBinarizer()
label_list = lb.fit_transform(label_list)
# print(label_list)

#split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(image_list,
                                                    label_list,
                                                    test_size=0.1,
                                                    random_state=42)



clients = FL_utils.create_clients_non_iid(X_train, y_train)

print(clients.keys())

clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = FL_utils.batch_data(data)


test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))



lr = 0.01
comms_round = 100
loss='categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(lr=lr,
                decay=lr / comms_round,
                momentum=0.9
               )



# initialize global model
smlp_global = FL_utils.SimpleMLP()
global_model = smlp_global.build(784, 10)

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
    for client in client_names:
        smlp_local = FL_utils.SimpleMLP()
        local_model = smlp_local.build(784, 10)
        local_model.compile(loss=loss,
                            optimizer=optimizer,
                            metrics=metrics)

        # set local model weight to the weight of the global model
        local_model.set_weights(global_weights)

        # fit local model with client's data
        local_model.fit(clients_batched[client], epochs=1, verbose=0)

        # scale the model weights and add to list
        scaling_factor = FL_utils.weight_scalling_factor(clients_batched, client)
        scaled_weights = FL_utils.scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)

        # clear session to free memory after each communication round
        K.clear_session()

    # to get the average over all the local model, we simply take the sum of the scaled weights
    average_weights = FL_utils.sum_scaled_weights(scaled_local_weight_list)

    # update global model
    global_model.set_weights(average_weights)

    # test global model and print out metrics after each communications round
    for (X_test, Y_test) in test_batched:
        global_acc, global_loss = FL_utils.test_model(X_test, Y_test, global_model, comm_round)

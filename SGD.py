
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



#create clients
clients = FL_utils.create_clients(X_train, y_train, num_clients=10)
# print(clients["client_3"])
# print(type(clients["client_3"]))
# print(clients["client_3"][1])



# process and batch the training data for each client
clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = FL_utils.batch_data(data)

# process and batch the test set
test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))



lr = 0.01
comms_round = 100
loss='categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(lr=lr,
                decay=lr / comms_round,
                momentum=0.9
               )



SGD_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(320)
smlp_SGD = FL_utils.SimpleMLP()
SGD_model = smlp_SGD.build(784, 10)

SGD_model.compile(loss=loss,
              optimizer=optimizer,
              metrics=metrics)

# fit the SGD training data to model
_ = SGD_model.fit(SGD_dataset, epochs=100, verbose=0)

#test the SGD global model and print out metrics
for(X_test, Y_test) in test_batched:
        SGD_acc, SGD_loss = FL_utils.test_model(X_test, Y_test, SGD_model, 1)
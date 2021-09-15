import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer


def load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data


def prepareTrainTest(path):
    data = load(path)
    data = np.array(data)
    np.random.shuffle(data)
    size = len(data)
    train_size = int(0.8*size)
    inputs = []
    outputs = []
    for elem in data:
        inputs.append(elem[0]/255) #TOFIX: also divide x, y by 255
        outputs.append(elem[1])
    x_train = inputs[:train_size]
    y_train = outputs[:train_size]
    x_test = inputs[train_size:]
    y_test = outputs[train_size:]
    y_train = np.asarray(y_train).astype('float32').reshape((-1, 1))
    y_test = np.asarray(y_test).astype('float32').reshape((-1, 1))
    return x_train, y_train, x_test, y_test


class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("sigmoid"))
        return model


def bcr(TP, FP, TN, FN):
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
            X_test: test set data
            y_test: test set labels
            model: the model
            comm_round: number of total communication rounds
        return:
            acc: accuracy of the model
            loss: global loss of the model
    """
    predictions = model.predict(x_test)
    print(predictions)
    print(y_test)
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


def simpleSGD(X_train, y_train, X_test, y_test, lr = 0.01, comms_round = 100):
    ''' Simple SGD algorithm
            args:
                clients: dictionary of the clients and their data
                X_test: test set data
                y_test: test set labels
            returns:
                SGD_acc: the global accuracy after comms_round rounds
    '''

    test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))

    loss = 'binary_crossentropy'
    metrics = ['accuracy']
    optimizer = SGD(lr=lr,
                    decay=lr / comms_round,
                    momentum=0.9
                    )

    SGD_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(320)
    smlp_SGD = SimpleMLP()
    SGD_model = smlp_SGD.build(32771, 1)

    SGD_model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    # fit the SGD training data to model
    _ = SGD_model.fit(SGD_dataset, epochs=100, verbose=2)

    #test the SGD global model and print out metrics
    for(X_test, Y_test) in test_batched:
            SGD_acc = test_model(X_test, Y_test, SGD_model)

    return SGD_acc



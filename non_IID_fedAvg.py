import numpy as np
import FL_utils


def simpleFedAvgTest(num_clients = 10, num_classes = 2, frac = 1, bs = 32, epo = 1, lr = 0.01, comms_round = 100):
    """ Make a simple non-IID federated averaging test
            args:
                num_clients: the number of clients to be created
                num_classes: the number of classes that each client has
                frac: fraction of clients selected at each round
                bs: local mini-batch size
                epo: number of local epochs
                lr: learning rate
                comms_round: number of global communication round
                comm_round: number of total communication rounds
    """

    X_train, X_test, y_train, y_test = FL_utils.prepareTrainTest('trainingSet/trainingSet')

    # create clients
    clients = FL_utils.create_clients_non_iid(X_train, y_train, num_clients=num_clients, num_classes=num_classes)
    FL_utils.fedAvg(clients, X_test, y_test, frac = frac, bs = bs, epo = epo, lr = lr, comms_round = comms_round)


def cross_val_fedAvg(splitter=5, num_clients = 10, num_classes = 2, frac = 1, bs = 32, epo = 1, lr = 0.01, comms_round = 100):
    """ Make a splitter-fold cross-validation non-IID federated averaging test
            args:
                splitter: the number of folds for the cross-validation
                num_clients: the number of clients to be created
                num_classes: the number of classes that each client has
                frac: fraction of clients selected at each round
                bs: local mini-batch size
                epo: number of local epochs
                lr: learning rate
                comms_round: number of global communication round
                comm_round: number of total communication rounds
    """

    image_list, label_list = FL_utils.prepareAllData('trainingSet/trainingSet')
    size = len(label_list)
    total_accuracy = 0
    for i in range(splitter):
        split = int(size / splitter)
        spliti = int(i * (size / splitter))
        X_train1 = image_list[0:spliti]  # 0:61
        y_train1 = label_list[0:spliti]  # 0:61
        print("train 1: " + str(0) + " à " + str(spliti))

        X_test = image_list[spliti:spliti + split]  # 61:122
        y_test = label_list[spliti:spliti + split]  # 61:122
        print("test: " + str(spliti) + " à " + str(spliti + split))
        X_train2 = image_list[spliti + split:]
        y_train2 = label_list[spliti + split:]
        print("train 2: " + str(spliti + split) + " à " + "la fin")

        X_train = np.append(X_train1, X_train2, axis=0)
        y_train = np.append(y_train1, y_train2, axis=0)

        clients = FL_utils.create_clients_non_iid(X_train, y_train, num_clients=num_clients, num_classes=num_classes)

        local_accuracy = FL_utils.fedAvg(clients, X_test, y_test, frac = frac, bs = bs, epo = epo, lr = lr, comms_round = comms_round)
        total_accuracy += local_accuracy
    return total_accuracy / splitter
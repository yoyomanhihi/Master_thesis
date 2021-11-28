# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy


def history(history):
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('dice score by epoch')
    plt.ylabel('dice score')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss by epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def history_fedavg(train_accs, test_accs, clientsnbr):
    for i in range(clientsnbr):
        train_accs_client = []
        for j in range(i, len(train_accs), clientsnbr):
            train_accs_client.append(train_accs[j])
        print(train_accs_client)
        plt.plot(train_accs_client)
    plt.plot(test_accs)
    plt.title('dice score by epoch')
    plt.ylabel('dice score')
    plt.xlabel('epoch')
    plt.legend(['train client 1', 'train client 2', 'train client 3', 'test'], loc='upper left')
    plt.show()

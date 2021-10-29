import unet_utils as utils
import tensorflow as tf
from tensorflow import keras
import numpy as np

client_path = 'NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics/LUNG1-002'

def build_and_save():
    x_train, y_train, x_test, y_test = utils.prepareTrainTest('unet_dataset.pickle')
    x_train = np.reshape(x_train, (len(x_train), 512, 512, 1))
    y_train = np.reshape(y_train, (len(y_train), 512, 512, 1))
    #
    model = utils.simpleSGD_2d(x_train, y_train, x_test, y_test)
    #
    model.save('unet_model.h5')

    print(utils.segmentation_2d(model, client_path, 52))


def load_and_segment():

    model = keras.models.load_model('unet_model.h5', compile=False)

    # SGD_acc = utils.test_model(x_test, y_test, model)

    print(utils.segmentation_2d(model, client_path, 43))


build_and_save()
# load_and_segment()
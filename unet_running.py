import unet_utils as utils
import tensorflow as tf
from tensorflow import keras
import numpy as np

client_nbr = 422
img_nbr = 55
organ = "heart"
client_path = 'NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics/LUNG1-' + str(client_nbr)
mask_path = 'NSCLC-Radiomics/manifest-1603198545583/masks_' + str(organ) + "/LUNG1-" + str(client_nbr) + "/mask_" + str(img_nbr) + ".png"
image_path = 'NSCLC-Radiomics/manifest-1603198545583/images' + "/LUNG1-" + str(client_nbr) + "/image_" + str(img_nbr) + ".png"
# array_path = 'NSCLC-Radiomics/manifest-1603198545583/arrays/LUNG1-' + str(client_nbr) + '/array_' + str(img_nbr) + ".npy"

def build_and_save():
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.set_visible_devices([], 'GPU')

    # x_train, y_train, x_test, y_test = utils.prepareTrainTest('datasets/lungs_first250(261)_1of36_geq8400000.pickle')

    # utils.simpleSGD(x_train, y_train, x_test, y_test, epochs=200)
    utils.simpleSGD(None, None, None, None, epochs=200)

    # print(utils.segmentation_2d(model, client_path, 37, "tumor"))


def build_and_save_fedavg():
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    datasetpath1 = 'datasets/lungs_first250(261)_1of36_geq8400000_1.pickle'
    datasetpath2 = 'datasets/lungs_first250(261)_1of36_geq8400000_2.pickle'
    datasetpath3 = 'datasets/lungs_first250(261)_1of36_geq8400000_3.pickle'

    # datasetpath1 = 'datasets/smallfortest.pickle'
    # datasetpath2 = 'datasets/smallfortest.pickle'
    # datasetpath3 = 'datasets/smallfortest.pickle'

    listdatasetspaths = [datasetpath1, datasetpath2, datasetpath3]

    clients, x_test, y_test = utils.createClients(listdatasetspaths)

    utils.fedAvg(clients, x_test, y_test, patience=15)



def load_and_segment():
    model = keras.models.load_model('best_val_loss.h5', compile=False)

    # SGD_acc = utils.test_model(x_test, y_test, model)

    print(utils.segmentation_2d(model, client_path, mask_path, image_path, img_nbr, organ))



def load_and_evaluate():
    model = keras.models.load_model('models/fedAvg_lungs_first250(261)_1of36_geq8400000_90epochs.h5', compile=False)

    x_test, y_test = utils.prepareTrainingData('datasets/lungs_first250(261)_1of36_geq8400000.pickle')

    SGD_acc = utils.test_model(x_test, y_test, model)

    print("model dice score on test dataset: " + str(SGD_acc))

build_and_save()
# build_and_save_fedavg()
# load_and_segment()
# load_and_evaluate()
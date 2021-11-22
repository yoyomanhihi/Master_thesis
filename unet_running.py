import unet_utils as utils
import tensorflow as tf
from tensorflow import keras
import numpy as np

client_nbr = 124
img_nbr = 28
organ = "heart"
client_path = 'NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics/LUNG1-' + str(client_nbr)
mask_path = 'NSCLC-Radiomics/manifest-1603198545583/masks_' + str(organ) + "/LUNG1-" + str(client_nbr) + "/mask_" + str(img_nbr) + ".png"
array_path = 'NSCLC-Radiomics/manifest-1603198545583/arrays/LUNG1-' + str(client_nbr) + '/array_' + str(img_nbr) + ".npy"

def build_and_save():
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.set_visible_devices([], 'GPU')

    x_train, y_train = utils.prepareTrainingData('datasets/smallfortest.pickle')

    model = utils.simpleSGD(x_train, y_train, epochs=200)

    # model.save('unet_model_bigtumors_first50_50epochs.h5')

    # print(utils.segmentation_2d(model, client_path, 37, "tumor"))


def build_and_save_fedavg():

    datasetpath1 = 'unet_dataset_lungs_first10.pickle'
    datasetpath2 = 'unet_dataset_lungs_11-20.pickle'
    listdatasetspaths = [datasetpath1, datasetpath2]

    clients, x_test, y_test = utils.createClients(listdatasetspaths)

    model = utils.fedAvg(clients, x_test, y_test)

    model.save('fedAvg_model.h5')


def load_and_segment():

    model = keras.models.load_model('models/heart_first50_1of3_geq8000000.h5', compile=False)

    # SGD_acc = utils.test_model(x_test, y_test, model)

    print(utils.segmentation_2d(model, client_path, mask_path, array_path, img_nbr, organ))




build_and_save()
# build_and_save_fedavg()
# load_and_segment()
import unet_utils as utils
import tensorflow as tf
from tensorflow import keras
import numpy as np

client_nbr = 107
img_nbr = 43
organ = "esophagus"
client_path = 'NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics/LUNG1-' + str(client_nbr)
mask_path = 'NSCLC-Radiomics/manifest-1603198545583/masks_' + str(organ) + "/LUNG1-" + str(client_nbr) + "/mask_" + str(img_nbr) + ".png"
image_path = 'NSCLC-Radiomics/manifest-1603198545583/images' + "/LUNG1-" + str(client_nbr) + "/image_" + str(img_nbr) + ".png"
datasetpath = 'datasets/dataset_heart/'
datasetpath_fedAvg = 'datasets/dataset_heart_fedAvg/'

def build_and_save(datasetpath, epochs):
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.set_visible_devices([], 'GPU')

    utils.simpleSGD(datasetpath=datasetpath, epochs=epochs)


def build_and_save_fedavg(datasetpath, nbclients):
    physical_devices = tf.config.list_physical_devices('GPU')
    # print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    utils.fedAvg(datasetpath, 3, patience=15)



def load_and_segment(model_path):
    model = keras.models.load_model(model_path, compile=False)

    # SGD_acc = utils.test_model(x_test, y_test, model)

    print(utils.segmentation_2d(model, client_path, mask_path, image_path, img_nbr, organ))



def load_and_evaluate():
    model = keras.models.load_model('models/fedAvg_test_dataaugm.h5', compile=False)

    x_test, y_test = utils.prepareTrainingData('datasets/test_heart_60-100(341-395)_1of3_geq8000000.pickle')

    SGD_acc = utils.test_model(x_test, y_test, model)

    print("model dice score on test dataset: " + str(SGD_acc))

build_and_save(datasetpath=datasetpath, epochs=200)
# build_and_save_fedavg(datasetpath=datasetpath_fedAvg, nbclients=3)
# load_and_segment('models/fedAvg_test_dataaugm.h5')
# load_and_evaluate()
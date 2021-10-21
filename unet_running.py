import unet_utils as utils
import tensorflow as tf
from tensorflow import keras

def make_all():
    client_path = 'NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics/LUNG1-376'

    x_train, y_train, x_test, y_test = utils.prepareTrainTest('unet_dataset.pickle')

    model = utils.simpleSGD_2d(x_train, y_train, x_test, y_test)

    model.save('unet_model.h5')

    # return SGD_acc

make_all()
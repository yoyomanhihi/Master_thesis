import unet_utils
import unet_utils as utils
import tensorflow as tf
from tensorflow import keras
import sys

client_nbr = 422
img_nbr = 56
organ = "heart"
client_path = 'NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics/LUNG1-' + str(client_nbr)
mask_path = 'NSCLC-Radiomics/manifest-1603198545583/masks_' + str(organ) + "/LUNG1-" + str(client_nbr) + "/mask_" + str(img_nbr) + ".png"
image_path = 'NSCLC-Radiomics/manifest-1603198545583/images' + "/LUNG1-" + str(client_nbr) + "/image_" + str(img_nbr) + ".png"
datasetpath = 'datasets/dataset_heart/'
datasetpath_fedAvg = 'datasets/dataset_heart_fedAvg/'
name = sys.argv[1]
print('name: ' + str(name))

def build_and_save(datasetpath, epochs):
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.set_visible_devices([], 'GPU')

    utils.simpleSGD(datasetpath=datasetpath, epochs=epochs, name=name)


def build_and_save_fedavg(datasetpath, nbclients):
    physical_devices = tf.config.list_physical_devices('GPU')
    # print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    utils.fedAvg(datasetpath, 3, name=name, patience=10)



def load_and_segment(model_path):
    model = keras.models.load_model(model_path, compile=False)

    # SGD_acc = utils.test_model(x_test, y_test, model)

    print(utils.segmentation_2d(model, client_path, mask_path, image_path, img_nbr, organ))



def load_and_evaluate(datasetpath, model):
    model = keras.models.load_model(model, compile=False)

    optimizer = tf.keras.optimizers.Adam

    model.compile(optimizer=optimizer(), metrics = [unet_utils.dice_coef_loss, unet_utils.dice_coef_loss_ponderated])

    SGD_acc = utils.test_model(datasetpath, model)

    ponderated_dice = SGD_acc[2] / unet_utils.get_average_number_of_true_pixels(datasetpath)
    print('ponderated dice: ' + str(ponderated_dice))

    print("model dice score on test dataset: " + str(SGD_acc))

    # ponderated_acc = utils.test_model_ponderated(x_test, y_test, model)
    # print("ponderated dice: " + str(ponderated_acc))

build_and_save(datasetpath=datasetpath, epochs=100)
# build_and_save_fedavg(datasetpath=datasetpath_fedAvg, nbclients=3)
# load_and_segment('models/heart_no_dataaugm_21epochs.h5')
# load_and_evaluate('datasets/dataset_example', 'models/heart_big_dataaugm_15epochs.h5')
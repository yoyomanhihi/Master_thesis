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
# datasetpath = 'datasets/dataset_heart/'
# datasetpath_fedAvg = 'datasets/dataset_heart_fedAvg/'
name = sys.argv[1]

def build_and_save(datasetpath, epochs, name):
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.set_visible_devices([], 'GPU')

    utils.simpleSGD(datasetpath=datasetpath, epochs=epochs, name=name)


def build_and_save_fedavg(datasetpath, nbclients, name):
    physical_devices = tf.config.list_physical_devices('GPU')
    # print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    utils.fedAvg(datasetpath, nbclients, name=name, patience=10)



def load_and_segment(model_path):
    model = keras.models.load_model(model_path, compile=False)

    # SGD_acc = utils.test_model(x_test, y_test, model)

    print(utils.segmentation_2d(model, client_path, mask_path, image_path, img_nbr, organ))


def load_and_evaluate(datasetpath, model):
    model = keras.models.load_model(model, compile=False)

    optimizer = tf.keras.optimizers.Adam

    model.compile(optimizer=optimizer(), metrics = [unet_utils.dice_coef_loss, unet_utils.dice_coef_loss_ponderated])

    SGD_acc = utils.test_model(datasetpath, model)


    print('dice score: ' + str(-SGD_acc[1]))
    ponderated_dice = SGD_acc[2] / unet_utils.get_average_number_of_true_pixels(datasetpath)
    print('ponderated dice: ' + str(-ponderated_dice))


def get_individial_dice(datasetpath, model, nbclients=3):
    model = keras.models.load_model(model, compile=False)

    optimizer = tf.keras.optimizers.Adam

    model.compile(optimizer=optimizer(), metrics = [unet_utils.dice_coef_loss, unet_utils.dice_coef_loss_ponderated])

    total_dice = 0.
    total_ponderated = 0.

    for i in range(nbclients):
        dataset_client = datasetpath + '/' + str(i)
        SGD_acc = utils.test_model(dataset_client, model)

        print('dice score for client ' + str(i) + ': ' + str(-SGD_acc[1]))
        total_dice -= SGD_acc[1]

        ponderated_dice = SGD_acc[2] / unet_utils.get_average_number_of_true_pixels(dataset_client)
        print('ponderated dice: ' + str(-ponderated_dice))
        total_ponderated -= ponderated_dice

    mean_dice = total_dice/nbclients
    print('mean dice: ' + str(mean_dice))

    mean_ponderated = total_ponderated/nbclients
    print('mean ponderated dice: ' + str(mean_ponderated))


# build_and_save(datasetpath='datasets/dataset_heart_fedAvg/', epochs=100, name=name)
# build_and_save_fedavg(datasetpath='datasets/dataset_fedAvg_example', nbclients=3, name=name)
# load_and_segment('models/heart_no_dataaugm_21epochs.h5')
# load_and_evaluate('datasets/dataset_heart_fedAvg/1', 'models/ds1_heart_39epochs.h5')
# get_individial_dice(datasetpath='datasets/dataset_heart_fedAvg', model='fed_1.h5')
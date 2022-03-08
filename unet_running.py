import unet_segmentation as segmentation
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

def build_and_save(datasetpath, preloaded, epochs, name):
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.set_visible_devices([], 'GPU')

    utils.simpleSGD(datasetpath=datasetpath, preloaded=preloaded, epochs=epochs, name=name)


def build_and_save_fedavg(datasetpath, preloaded, nbclients, name):
    physical_devices = tf.config.list_physical_devices('GPU')
    # print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    utils.fedAvg(datasetpath, preloaded, nbclients, name=name, patience=10)



def load_and_segment(model_path):
    model = keras.models.load_model(model_path, compile=False)

    # SGD_acc = utils.test_model(x_test, y_test, model)

    print(segmentation.segmentation_2d(model, client_path, mask_path, image_path, img_nbr, organ))


def load_and_evaluate(datasetpath, model):
    model = keras.models.load_model(model, compile=False)

    optimizer = tf.keras.optimizers.Adam

    model.compile(optimizer=optimizer(), metrics = [utils.dice_coef_loss, utils.dice_coef_loss_ponderated])

    SGD_acc = utils.test_model(datasetpath, model)


    print('dice score: ' + str(-SGD_acc[1]))
    ponderated_dice = SGD_acc[2] / utils.get_average_number_of_true_pixels(datasetpath)
    print('ponderated dice: ' + str(-ponderated_dice))


def get_individial_dice_3d(datasetpath, model, nbclients=3):
    total_dice = 0.

    for i in range(nbclients):
        dataset_client = datasetpath + '/' + str(i)
        SGD_acc = utils.test_model_3d(dataset_client, model)

        print('3d dice score for client ' + str(i) + ': ' + str(SGD_acc))
        total_dice += SGD_acc

    mean_dice = total_dice/nbclients
    print('mean 3d dice: ' + str(mean_dice))


# build_and_save(datasetpath='datasets/dataset_example', epochs=3, name=name)
# build_and_save_fedavg(datasetpath='datasets/dataset_fedAvg_example', nbclients=3, name=name)
# load_and_segment('test.h5')
# load_and_evaluate('datasets/dataset_heart_fedAvg/1', 'ds1_new.h5')
# get_individial_dice_3d(datasetpath='datasets/dataset_heart_fedAvg', model='models/heart_fed_medbigda_27epochs(2).h5')
# plots.plot_from_file("ds1_data.txt", name="marchestp")
# plots.compare_fedAvg_to_separate_models("data/ds0_heart_20epochs(2).txt", "data/heart_fed_medbigda_27epochs(2).txt", 0, 3, "graph_0")
# print(utils.test_model_3d('datasets/dataset_heart_fedAvg/2', 'custom_loss_ds2.h5'))
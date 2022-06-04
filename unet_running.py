import unet_segmentation as segmentation
import unet_utils as utils
import tensorflow as tf
from tensorflow import keras
import sys

name = sys.argv[1]

def build_and_save(datasetpath, preloaded, epochs, name):
    """ Builds and saves a model using the classical centralised approach
    Args:
        datasetpath(string): path to the dataset
        preloaded(string): name of the preloaded model. If None, the model is built from scratch
        epochs(int): maximal number of epochs to train the model
        name: name of the model to save
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.set_visible_devices([], 'GPU')

    utils.simpleSGD(datasetpath=datasetpath, preloaded=preloaded, epochs=epochs, name=name)


def build_and_save_fedeq(datasetpath, preloaded, nbclients, name):
    """ Builds and saves a model using the federated equal-chances approach
    Args:
        datasetpath(string): path to the dataset
        preloaded(string): name of the preloaded model. If None, the model is built from scratch
        nbclients(int): number of clients to use
        name: name of the model to save
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    # print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    utils.fedEq(datasetpath, preloaded, nbclients, name=name, patience=10)


def build_and_save_fedavg_original(datasetpath, preloaded, nbclients, name):
    """ Builds and saves a model using the federated averaging approach
    Args:
        datasetpath(string): path to the dataset
        preloaded(string): name of the preloaded model. If None, the model is built from scratch
        nbclients(int): number of clients to use
        name: name of the model to save
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    # print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    utils.fedAvg(datasetpath, preloaded, nbclients, name=name, patience=10)


def load_and_segment(model_path, client_path, mask_path, image_path, img_nbr, organ):
    """ Loads a model and segments an image
    Args:
        model_path(string): path to the model
        client_path(string): path to the client folder
        mask_path(string): path to the mask folder
        image_path(string): path to the image folder
        img_nbr(int): number of the image to segment
        organ(string): name of the organ to segment
    """
    model = keras.models.load_model(model_path, compile=False)

    # SGD_acc = utils.test_model(x_test, y_test, model)

    print(segmentation.segmentation_2d(model, client_path, mask_path, image_path, img_nbr, organ))


def load_and_evaluate(datasetpath, model):
    """ Loads a model and evaluates it on the test set
    Args:
        datasetpath(string): path to the dataset
        model(string): name of the model to evaluate
    """
    model = keras.models.load_model(model, compile=False)

    optimizer = tf.keras.optimizers.Adam

    model.compile(optimizer=optimizer(), metrics = [utils.dice_coef_loss, utils.dice_coef_loss_ponderated])

    SGD_acc = utils.test_model(datasetpath, model)


    print('dice score: ' + str(-SGD_acc[1]))
    ponderated_dice = SGD_acc[2] / utils.get_average_number_of_true_pixels(datasetpath)
    print('ponderated dice: ' + str(-ponderated_dice))


def get_individial_dice_3d(datasetpath, model, nbclients=3):
    """ Loads a model and evaluates its 3d dice score on the test set
    Args:
        datasetpath(string): path to the dataset
        model(string): name of the model to evaluate
        nbclients(int): number of clients to use
    """
    total_dice = 0.
    for i in range(nbclients):
        dataset_client = datasetpath + '/' + str(i)
        SGD_acc = utils.test_model_3d(dataset_client, model)

        print('3d dice score for client ' + str(i) + ': ' + str(SGD_acc))
        total_dice += SGD_acc

    mean_dice = total_dice/nbclients
    print('mean 3d dice: ' + str(mean_dice))

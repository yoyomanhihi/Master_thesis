# Visualize training history
import os
import numpy as np
import matplotlib.pyplot as plt
import file_utils
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
import unet_preprocessing
import unet_segmentation
import tensorflow as tf

# Mean  intensity and standard deviation of the pixels in the image
MEAN = 4611.838943481445
STD = 7182.589254997573

def history(train_accs, val_accs, name):
    """ Plot the history of the training and validation accuracy
    Args:
        train_accs (list): list of training accuracies
        val_accs (list): list of validation accuracies
        name (str): name of the plot
    """
    plot_name = name + '.png'

    # make the x axis start to 1
    x_axis = range(1, len(train_accs)+1)

    # make the plot
    plt.plot(x_axis, train_accs)
    plt.plot(x_axis, val_accs)
    plt.title("Dice's coefficient by epoch")
    plt.ylabel("Dice's coefficient")
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(plot_name)
    plt.show()


def history_fedavg(train_accs, val_accs, clientsnbr, name):
    """ Plot the history of the training and validation accuracy
        for each client, in the federated average case
    Args:
        train_accs (list): list of training accuracies
        val_accs (list): list of validation accuracies
        clientsnbr (int): number of clients
        name (str): name of the plot
    """

    plot_name = name + '.png'
    colors = ['blue', 'red', 'green']

    # make the x axis start to 1
    x_axis = range(1, len(train_accs) + 1)

    # plot the train accuracies of the clients
    for i in range(clientsnbr):
        train_accs_client = []
        for j in range(i, len(train_accs), clientsnbr):
            train_accs_client.append(train_accs[j])
        print(train_accs_client)
        plt.plot(x_axis, train_accs_client, color=colors[i])

    # plot the validation accuracies of the clients
    for i in range(clientsnbr):
        val_accs_client = []
        for j in range(i, len(val_accs), clientsnbr):
            val_accs_client.append(val_accs[j])
        print(val_accs_client)
        plt.plot(x_axis, val_accs_client, color=colors[i], linestyle='dotted')

    # make the plot
    plt.title("Dice's coefficient by epoch")
    plt.ylabel("Dice's coefficient")
    plt.xlabel('epoch')
    plt.legend(['train client 1', 'train client 2', 'train client 3', 'validation client 1',
                'validation client 2', 'validation client 3'], loc='lower right')
    plt.savefig(plot_name)
    plt.show()


def plot_from_file(filepath, name):
    """ Plot the history of the training and validation accuracy
        from the file containing the history
    Args:
        filepath (str): path to the file containing the history
        name (str): name of the plot
    """

    plot_name = name + '.png'

    # read the file
    lines = file_utils.read_measures(filepath)

    # divide the lines in train and validation accuracies
    mid = int(len(lines)/2)
    print("epoch optimal: " + str(mid-10))
    train = lines[:mid]
    val = lines[mid:]

    # make the plot
    history(train, val, plot_name)


def interpret_fed_path(fed_path, dataset_nbr, nbclients):
    lines = file_utils.read_measures(fed_path)
    separation = int(len(lines)/7)
    train = lines[0:separation*3] # Train acc of the three clients
    train_client = []

    # Takes only the train of the interesting dataset
    for i in range(dataset_nbr, len(train), nbclients):
        train_client.append(train[i])

    val = lines[separation*3:separation*6]
    val_client = []

    # Takes only the train of the interesting dataset
    for i in range(dataset_nbr, len(val), nbclients):
        val_client.append(val[i])

    return train_client, val_client



def compare_fedAvg_to_separate_models(local_path, glob_path, fedor_path, fedeq_path,
                                      client_nbr, nbclients, name, smooth, step):
    """ Plot the training and validation accuracy of all strategies
        from the files containing the history
    Args:
        local_path (str): path to the file containing the history of the local strategy
        glob_path (str): path to the file containing the history of the global strategy
        fedor_path (str): path to the file containing the history of the original fedAvg strategy
        fedeq_path (str): path to the file containing the history of the federated equal-chances strategy
        client_nbr (int): number of the client
        nbclients (int): total number of clients
        name (str): name of the plot
        smooth (int): value of the smooth parameter used
        step (int): step of the strategy (1 or 2)
    """


    plot_name = name + '.png'

    # local model
    local_lines = file_utils.read_measures(local_path)
    local_mid = int(len(local_lines) / 2)
    local_train = local_lines[:local_mid]
    local_val = local_lines[local_mid:]
    x_axis1 = range(1, len(local_train) + 1)
    plt.plot(x_axis1, local_train, color='blue')
    plt.plot(x_axis1, local_val, color='blue', linestyle='dotted', label='_nolegend_')

    # global model
    local_lines = file_utils.read_measures(glob_path)
    local_mid = int(len(local_lines) / 2)
    local_train = local_lines[:local_mid]
    local_val = local_lines[local_mid:]
    x_axis1 = range(1, len(local_train) + 1)
    plt.plot(x_axis1, local_train, color='orange')
    plt.plot(x_axis1, local_val, color='orange', linestyle='dotted', label='_nolegend_')

    # fedor
    fed_train, fed_val = interpret_fed_path(fedor_path, client_nbr, nbclients)
    x_axis2 = range(1, len(fed_train) + 1)
    plt.plot(x_axis2, fed_train, color='red')
    plt.plot(x_axis2, fed_val, color='red', linestyle='dotted', label='_nolegend_')

    # fedeq
    fedeq_train, fedeq_val = interpret_fed_path(fedeq_path, client_nbr, nbclients)
    x_axis3 = range(1, len(fedeq_train) + 1)
    plt.plot(x_axis3, fedeq_train, color='green')
    plt.plot(x_axis3, fedeq_val, color='green', linestyle='dotted', label='_nolegend_')

    # make the plot
    plt.title("Strategies training details of step " + str(step) +  " for dataset " + str(client_nbr+1))
    plt.ylabel("Dice's coefficient, smooth = " + str(smooth))
    plt.xlabel('epoch')
    # plt.legend(['local train', 'local validation', 'global train', 'global validation', 'original fedAvg train', 'original fedAvg validation', 'equal-chances fedAvg train', 'equal-chances fedAvg validation'], loc='lower right')
    plt.legend(['local model', 'global model', 'original fedAvg', 'equal-chances fedAvg'],
               loc='lower right')
    plt.savefig(plot_name)
    plt.show()


def mask_3d(mask_path, patient_nbr, title):
    """ Make the 3d plot of the original mask
    Args:
        mask_path (str): path to the file containing the masks
        patient_nbr (int): number of the patient
        title (str): title of the plot
    """

    patient = []
    listmasks = os.listdir(mask_path)
    # sort the masks
    sortedmasks = unet_preprocessing.sorted_alphanumeric(listmasks)

    # create the 3d array of the masks
    for mask in sortedmasks:
        nbr = int(mask.split('_')[0])
        if nbr == patient_nbr:
            path = mask_path + '/' + mask
            mask_2d = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            patient.append(mask_2d)
    patient = np.array(patient)

    # make the plot
    plot_3d(patient, title)


def prediction_3d(images_path, model, patient_nbr, title):
    """ Make the 3d plot of the predicted mask
    Args:
        images_path (str): path to the file containing the images
        model (keras model): model used to predict the masks
        patient_nbr (int): number of the patient
        title (str): title of the plot
    """

    # load the model
    model = tf.keras.models.load_model(model, compile=False)

    patient = []
    listimages = os.listdir(images_path)
    # sort the images
    sortedimages = unet_preprocessing.sorted_alphanumeric(listimages)

    # create the 3d array of the predicted masks
    for image in sortedimages:
        nbr = int(image.split('_')[0])
        if nbr == patient_nbr:
            image_path = images_path + '/' + image
            prediction = unet_segmentation.prediction(image_path, model)
            patient.append(prediction)
    patient = np.array(patient)

    # make the plot
    plot_3d(patient, title)


def plot_3d(image, title, threshold=0.5, color="navy"):
    """ Make the 3d plot
    Args:
        image (numpy array): 3d array of the image to plot
        title (str): title of the plot
        threshold (float): threshold to binarize the image
        color (str): color of the plot
    """

    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces, _, _ = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.2)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.title(title, fontsize=25)
    plt.show()
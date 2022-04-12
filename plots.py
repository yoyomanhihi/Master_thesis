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


MEAN = 4611.838943481445
STD = 7182.589254997573


def history(train_accs, val_accs, name):

    plot_name = name + '.png'

    x_axis = range(1, len(train_accs)+1)

    plt.plot(x_axis, train_accs)
    plt.plot(x_axis, val_accs)
    plt.title("Dice's coefficient by epoch")
    plt.ylabel("Dice's coefficient")
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(plot_name)
    plt.show()


def history_fedavg(train_accs, val_accs, clientsnbr, name):

    plot_name = name + '.png'
    colors = ['blue', 'red', 'green']

    x_axis = range(1, len(train_accs) + 1)

    for i in range(clientsnbr):
        train_accs_client = []
        for j in range(i, len(train_accs), clientsnbr):
            train_accs_client.append(train_accs[j])
        print(train_accs_client)
        plt.plot(x_axis, train_accs_client, color=colors[i])
    for i in range(clientsnbr):
        val_accs_client = []
        for j in range(i, len(val_accs), clientsnbr):
            val_accs_client.append(val_accs[j])
        print(val_accs_client)
        plt.plot(x_axis, val_accs_client, color=colors[i], linestyle='dotted')
    plt.title("Dice's coefficient by epoch")
    plt.ylabel("Dice's coefficient")
    plt.xlabel('epoch')
    plt.legend(['train client 1', 'train client 2', 'train client 3', 'validation client 1', 'validation client 2', 'validation client 3'], loc='lower right')
    plt.savefig(plot_name)
    plt.show()


def plot_from_file(filepath, name):
    plot_name = name + '.png'

    lines = file_utils.read_measures(filepath)
    mid = int(len(lines)/2)
    print("epoch optimal: " + str(mid-10))
    train = lines[:mid]
    val = lines[mid:]
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



def compare_fedAvg_to_separate_models(local_path, glob_path, fedor_path, fedeq_path, client_nbr, nbclients, name, smooth, step):
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


    plt.title("Strategies training details of step " + str(step) +  " for dataset " + str(client_nbr+1))
    plt.ylabel("Dice's coefficient, smooth = " + str(smooth))
    plt.xlabel('epoch')
    # plt.legend(['local train', 'local validation', 'global train', 'global validation', 'original fedAvg train', 'original fedAvg validation', 'equal-chances fedAvg train', 'equal-chances fedAvg validation'], loc='lower right')
    plt.legend(['local model', 'global model', 'original fedAvg', 'equal-chances fedAvg'],
               loc='lower right')

    plt.savefig(plot_name)
    plt.show()


def mask_3d(mask_path, patient_nbr, title):
    patient = []
    listmasks = os.listdir(mask_path)
    sortedmasks = unet_preprocessing.sorted_alphanumeric(listmasks)
    for mask in sortedmasks:
        nbr = int(mask.split('_')[0])
        if nbr == patient_nbr:
            path = mask_path + '/' + mask
            mask_2d = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            patient.append(mask_2d)
    patient = np.array(patient)
    plot_3d(patient, title)


def prediction_3d(images_path, model, patient_nbr, title):
    model = tf.keras.models.load_model(model, compile=False)
    patient = []
    listimages = os.listdir(images_path)
    sortedimages = unet_preprocessing.sorted_alphanumeric(listimages)
    for image in sortedimages:
        nbr = int(image.split('_')[0])
        if nbr == patient_nbr:
            image_path = images_path + '/' + image
            prediction = unet_segmentation.prediction(image_path, model)
            patient.append(prediction)
    patient = np.array(patient)
    print(np.max(patient))
    print(np.min(patient))
    print(np.shape(patient))
    plot_3d(patient, title)


def plot_3d(image, title, threshold=0.5, color="navy"):
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

# compare_fedAvg_to_separate_models('data/ds0_heart_20epochs(2).txt',
#                                   'data/glo_final_0_data.txt',
#                                   'data/fedor_final_0_3_data.txt',
#                                   'data/fed2_0_data.txt',
#                                   0, 3, 'on_verra', 1, 1)

# compare_fedAvg_to_separate_models('data/ds0_medbig_data.txt',
#                                   'data/glo_final_1_data.txt',
#                                   'data/fed_or_data.txt',
#                                   'data/fed_50_data.txt',
#                                   0, 3, 'on_verra', 5000, 2)
#
# compare_fedAvg_to_separate_models('data/ds1_medbig_data.txt',
#                                   'data/glo_final_1_data.txt',
#                                   'data/fed_or_data.txt',
#                                   'data/fed_50_data.txt',
#                                   1, 3, 'on_verra', 5000, 2)
#
# compare_fedAvg_to_separate_models('data/ds2_medbig_data.txt',
#                                   'data/glo_final_1_data.txt',
#                                   'data/fed_or_data.txt',
#                                   'data/fed_50_data.txt',
#                                   2, 3, 'on_verra', 5000, 2)
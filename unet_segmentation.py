import numpy as np
import dcm_contour
import os
import seaborn
import matplotlib.pyplot as plt
import cv2
import imageio
import unet_utils

# Mean and std of the pixels in the dataset
MEAN = 4611.838943481445
STD = 7182.589254997573


def prediction(image_path, model):
    """ Use the model to predict the organ in the image
    Args:
        image_path: path to the image
        model: the model to use
    Return:
        prediction: the binary prediction of the organ
    """
    image = imageio.imread(image_path)
    image[0][0] = 0
    image = image - MEAN
    image = image / STD
    image = np.reshape(image, (1, 512, 512, 1))
    prediction = model.predict(image)
    prediction[prediction > 0.5] = 1
    prediction[prediction <= 0.5] = 0
    prediction = prediction.reshape(512, 512)
    return prediction


def show_rtstruct(organ, dcm_path, img_nbr):
    """ Show the contour of the tumor in the image as stored
        in the RTSTRUCT format
    Args:
        organ: the organ to be segmented (works for "esophagus", "heart" and "lung")
        dcm_path: path to the dicom file
        img_nbr: the number of the image to be segmented
    """
    if organ == "esophagus":
        # plot the rt struct of the image
        index = dcm_contour.get_index(dcm_path, "Esophagus")
        images, contours, _ = dcm_contour.get_data(dcm_path, index=index)
        for img_arr, contour_arr in zip(images[img_nbr:img_nbr+1], contours[img_nbr:img_nbr+1]):
            dcm_contour.plot2dcontour(img_arr, contour_arr, img_nbr)
        contours = np.array(contours)
        cntr = contours[img_nbr, 0]
    elif organ == "heart":
        index = dcm_contour.get_index(dcm_path, "Heart")
        images, contours, _ = dcm_contour.get_data(dcm_path, index=index)
        for img_arr, contour_arr in zip(images[img_nbr:img_nbr+1], contours[img_nbr:img_nbr+1]):
            dcm_contour.plot2dcontour(img_arr, contour_arr, img_nbr)
        cntr = contours[img_nbr][0]
        for i in range(1, len(contours[img_nbr])):
            cntr += contours[img_nbr][i]
    elif organ == "lung":
        index1 = dcm_contour.get_index(dcm_path, "Lung-Left")
        index2 = dcm_contour.get_index(dcm_path, "Lung-Right")
        images, contours1, _ = dcm_contour.get_data(dcm_path, index=index1)
        images2, contours2, _ = dcm_contour.get_data(dcm_path, index=index2)
        for i in range(len(contours1)):
            contours1[i] = np.append(contours1[i], contours2[i], axis=0)
        for img_arr, contour_arr in zip(images[img_nbr:img_nbr + 1], contours1[img_nbr:img_nbr + 1]):
            dcm_contour.plot2dcontour(img_arr, contour_arr, img_nbr)
        cntr = contours1[img_nbr][0]
        for i in range(1, len(contours1[img_nbr])):
            cntr += contours1[img_nbr][i]
    plt.show()
    return cntr


def heatMap(prediction):
    """ Plot the heat map of the prediction
    Args:
        prediction: the non-binary prediction
    """
    fig, ax = plt.subplots()
    # title = "Prediction heatmap"
    # plt.title(title, fontsize=18)
    # ttl = ax.title
    # ttl.set_position([0.5, 1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    seaborn.heatmap(prediction, ax=ax)
    plt.show()


def showFinalPrediction(cntr, prediction):
    """ Show the final prediction of the organ and its true contour
    Args:
        cntr: the true contour of the organ
        prediction: the prediction of the organ
    """
    for i in range(len(prediction)):
        for j in range(len(prediction[i])):
            if prediction[(i, j)] >= 0.5 and cntr[(i, j)] != 1:
                cntr[(i, j)] = 10.0
            elif cntr[(i, j)] == 1:
                cntr[(i, j)] = 50.0
    plt.imshow(cntr, cmap='magma')
    plt.show()



def segmentation_2d(model, client_path, mask_path, image_path, img_nbr, organ):
    """ Process the 2d segmentation of an image and plot the heatmap of the tumor predictions
    Args:
        model: the model to use
        client_path: path to the client folder
        mask_path: path to the mask folder
        image_path: path to the image folder
        img_nbr: the number of the image to be segmented
        organ: the organ to be segmented (works for "esophagus", "heart" and "lung")
    Returns:
        cntr: the true contour of the organ
        prediction: the prediction mask of the organ
    """
    dcm_file0 = os.listdir(client_path)[0]
    dcm_path0 = client_path + "/" + dcm_file0
    dcm_files = os.listdir(dcm_path0)
    for file in dcm_files:
        dcm_path = dcm_path0 + "/" + file
        if len(os.listdir(dcm_path)) > 5:
            break

    cntr = show_rtstruct(organ, dcm_path, img_nbr)

    image = imageio.imread(image_path)
    image[0][0] = 0
    image = image-MEAN
    image = image/STD
    image = np.reshape(image, (1, 512, 512, 1))
    predictions = model.predict(image)
    predictions = np.reshape(predictions, (512, 512))

    heatMap(predictions)

    showFinalPrediction(cntr, predictions)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask[mask <= 0.5] = 0  # Set out of tumor to 0
    mask[mask > 0.5] = 1  # Set out of tumor to 1
    dice = unet_utils.dice_coef_2(mask, predictions)
    print("dice accuracy: " + str(dice))

    return cntr, predictions
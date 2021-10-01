import cv2
import sys
import numpy as np
import os
from tensorflow.keras.layers import Flatten
import pickle
import re
import random
import pydicom as dicom

general_path = "NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics"
# general_path = "NSCLC-Radiomics-Interobserver1/NSCLC-Radiomics-Interobserver1"

MEAN = 0.1783
STD = 0.0844
MEANXY = 0.5
STDXY = 0.28
MEANZ = 870.6
STDZ = 454.638


def getMeanAndStd(general_path, nbclients):
    """ Get the mean and the standard deviation of the pixels from the clients.
        Useful for whitening
        args:
            general_path: path to all clients
            nbclients: number of clients to be evaluated to calculate the mean and standard deviation
    """
    allclients = []
    files = os.listdir(general_path)
    files.sort()
    for f in files[:nbclients]:
        if f != 'LICENSE':
            newpath = general_path + "/" + f
            images_path = newpath + "/images"
            images_files = os.listdir(images_path)
            for i in range(len(images_files)):
                image_file = images_path + "/image_" + str(i) + ".png"
                image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                image = image/255
                allclients.extend(image)
    allclients = np.array(allclients)
    print("mean: " + str(allclients.mean()))
    print("std: " + str(allclients.std()))



def generateImagesPath(general_path, client_nbr):
    """ Generate path to the images of the client
        args:
            general_path: path to all clients
            client_nbr: number of the client from which we want the path
        return:
            path to the images of the client
    """
    file = os.listdir(general_path)[client_nbr+1]
    path = general_path + "/" + file
    return path + "/images"


def generateMasksPath(general_path, client_nbr):
    """ Generate path to the masks of the client
        args:
            general_path: path to all clients
            client_nbr: number of the client from which we want the path
        return:
            path to the masks of the client
    """
    file = os.listdir(general_path)[client_nbr+1]
    path = general_path + "/" + file
    return path + "/masks"

images_path = generateImagesPath(general_path, 0)
masks_path = generateMasksPath(general_path, 0)


def displayImage(image):
    """ Display the 2d image"""
    cv2.imshow('Window name', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def isYellow(pixel): #yellow = 215, purple = 30
    ''' Return True if the pixel of the mask is yellow (if it is in the segmented zone) '''
    if pixel == 215:
        return True


def crop(image, y, x): #vertical, horizontal
    ''' Return a 32x32 image with top left corner of coordonate(y, x) '''
    crop_img = image[y:y + 32, x:x + 32]
    return crop_img


def isFullYellow(mask, y, x):
    ''' Return True if the 4 corners of the 32x32 image are yellow '''
    if isYellow(mask[y, x]) and isYellow(mask[y + 32, x]) and isYellow(mask[y, x + 32]) and isYellow(mask[y + 32, x + 32]):
        return True


def allFullYellow(mask, jump=2):
    ''' Return a list of coordonates of the image that are full yellow
        args:
            mask: the image with the segmented tumor
            jump: the jump in the range'''
    allcoords = []
    for y in range(0, 480, jump):
        for x in range(0, 480, jump):
            if (isFullYellow(mask, y, x)):
                allcoords.append((y, x))
    return allcoords


def isFullPurple(mask, y, x):
    ''' Return True if the 4 corners of the 32x32 image are purple '''
    if not (isYellow(mask[y, x]) or isYellow(mask[y + 32, x]) or isYellow(mask[y, x + 32]) or isYellow(mask[y + 32, x + 32])):
        return True


def allFullPurple(mask, jump=200):
    ''' Return a list of coordonates of the image that are full purple
        args:
            mask: the image with the segmented tumor
            jump: the jump in the range'''
    allcoords = []
    for y in range(0, 480, jump):
        for x in range(0, 480, jump):
            if (isFullPurple(mask, y, x)):
                allcoords.append((y, x))
    return allcoords


def isMostlyYellow(mask, y, x):
    ''' Return True if more than 50% of the 32x32 image is yellow '''
    sub_image = crop(mask, y, x)
    return np.sum(sub_image) > 125440 #512*215 + 512*30


def isMostlyPurple(mask, y, x):
    sub_image = crop(mask, y, x)
    somme = np.sum(sub_image)
    return somme > 30720 and somme < 125440


def allMostlyYellow(mask, jump=7):
    ''' Return a list of coordonates of the image that are more than 50% yellow
        args:
            mask: the image with the segmented tumor
            jump: the jump in the range'''
    mostly_yellows = []
    mostly_purples= []
    for y in range(0, 480, jump):
        for x in range(0, 480, jump):
            if (isMostlyYellow(mask, y, x)):
                mostly_yellows.append((y, x))
            elif (isMostlyPurple(mask, y, x)):
                if random.randint(0,3) == 0:
                    mostly_purples.append((y, x))
    return mostly_yellows, mostly_purples


def randomFullPurple(mask, nbr = 1):
    ''' Generate at most nbr random coordonates of full purple coordonates
        args:
            mask: the 512 x 512 mask of the segmentation
            nbr: the number of random pixel generated (kept only if full purple)
        returns:
            purples, the list of full purple pixels coordonates'''
    purples = []
    for i in range(nbr):
        y = random.randint(0, 479)
        x = random.randint(0, 479)
        if isFullPurple(mask, y, x):
            purples.append((y, x))
    return purples


def prepareAllYellows(allyellows, image, z):
    ''' Prepare the data for all tumor subimages from one 2d image
        args:
            allyellows: list of top left corner pixels from image to select
            image: the 2d image
        returns:
            data: the list of subimages + label to add to the dataset'''
    images_list = []
    for pixel in allyellows:
        flatten_subimage = crop(image, pixel[0], pixel[1]).flatten()
        flatten_subimage = np.append(flatten_subimage, (z - MEANZ) / STDZ)
        flatten_subimage = np.append(flatten_subimage, (pixel[0]-MEANXY)/STDXY)
        flatten_subimage = np.append(flatten_subimage, (pixel[1]-MEANXY)/STDXY)
        images_list.append(flatten_subimage)
    labels_list = np.ones((len(allyellows),), dtype=int)
    data = list(zip(images_list, labels_list))
    return data


def prepareAllPurples(allpurples, image, z):
    ''' Prepare the data for all non-tumor subimages from one 2d image
        args:
            allyellows: list of top left corner pixels from image to select
            image: the 2d image
        returns:
            data: the list of subimages + label to add to the dataset'''
    images_list = []
    for pixel in allpurples:
        flatten_subimage = crop(image, pixel[0], pixel[1]).flatten()
        flatten_subimage = np.append(flatten_subimage, (z - MEANZ) / STDZ)
        flatten_subimage = np.append(flatten_subimage, (pixel[0]-MEANXY)/STDXY)
        flatten_subimage = np.append(flatten_subimage, (pixel[1]-MEANXY)/STDXY)
        images_list.append(flatten_subimage)
    labels_list = np.zeros((len(allpurples),), dtype=int)
    data = list(zip(images_list, labels_list))
    return data



def sorted_alphanumeric(data):
    """ Sort the data alphanumerically. allows to have mask_2 before mask_10 for example"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)



def getZ(dcm_path):
    """ Get the slice_thickness and the initial position of z in order to compute the z position
        args:
            dcm_path: path to the dcm files
        return:
            slice_thickness: the thickness between two slices
    """
    slices = [dicom.dcmread(dcm_path + '/' + s) for s in os.listdir(dcm_path)[1:]]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))  # Sort by z axis
    print(dcm_path)
    try:
        slice_thickness = slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2]
        Z0 = slices[0].ImagePositionPatient[2]
    except:
        slice_thickness = slices[1].SliceLocation - slices[0].SliceLocation
        Z0 = slices[0].SliceLocation
    return slice_thickness, Z0



def generateDatasetFromOneClient(masks_path, images_path, dcm_path):
    ''' Generate the full 2d dataset from one client
        args:
            masks_path: directory path to all the masks of the client
            images_path: directory path to all images of the client
        returns:
            dataset: the full dataset of the client'''
    dataset = []
    masks_files = sorted_alphanumeric(os.listdir(masks_path))
    slice_thickness, Z0 = getZ(dcm_path)
    count0 = 0
    count1 = 0
    for i in range(len(masks_files)):
        mask_file = masks_path + "/mask_" + str(i) + ".png"
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        image_file = images_path + "/image_" + str(i) + ".png"
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        image = image/255
        image = image-MEAN
        image = image/STD
        allyellows, allpurples = allMostlyYellow(mask)
        allpurples.extend(randomFullPurple(mask))
        count0 += len(allpurples)
        count1 += len(allyellows)
        z = Z0 + i*slice_thickness
        dataset.extend(prepareAllYellows(allyellows, image, z))
        dataset.extend(prepareAllPurples(allpurples, image, z))
    print("count0: " + str(count0))
    print("count1: " + str(count1))
    return dataset


def evaluateDatasetRatio(dataset):
    ''' Count the repartition of tumors and non tumor examples in the dataset
        args:
            dataset: dataset of images and their classification
        return:
            count0: number of non tumor examples
            count1: number of tumor examples'''
    count0 = 0
    count1 = 0
    for i in dataset:
        if i[1] == 0:
            count0 += 1
        elif i[1] == 1:
            count1 += 1
        else:
            print("problem")
    print("count0: " + str(count0))
    print("count1: " + str(count1))
    return (count0, count1)


def storeDataset(dataset, name):
    """ Store the dataset as a pickle file"""
    with open(name, 'wb') as output:
        pickle.dump(dataset, output)


def generateDatasetFromManyClients(general_path, nbclients = 300):
    ''' Generate a dataset with example images from many clients
        args:
            general_path: path to all the client's images and masks
            nbclients: number of clients to be considered to create the dataset
        return:
            dataset: the final dataset generated'''
    dataset = []
    files = os.listdir(general_path)
    files.sort()
    size = min(nbclients, len(files)-2)
    for f in files[:size]:
        if f != 'LICENSE':
            newpath = general_path + "/" + f
            images_path = newpath + "/images"
            masks_path = newpath + "/masks"
            dcm_file = os.listdir(newpath)[0]
            dcm_path = newpath + "/" + dcm_file
            dcm_files2 = os.listdir(dcm_path)
            for dcm_file2 in dcm_files2:
                dcm_path2 = dcm_path + "/" + dcm_file2
                if len(os.listdir(dcm_path2)) > 5:
                    break
            dataset.extend(generateDatasetFromOneClient(masks_path, images_path, dcm_path2))
    return dataset


def generateAndStore(name, nbclients):
    ''' Generate a dataset from many clients and store it in the files
        args:
            name: name of the file to save
            nbclients: number of clients to be considered to create the dataset
        return:
            evalutation: tuple of the form (count0, count1)
            count0: number of non tumor examples
            count1: number of tumor examples'''
    dataset = generateDatasetFromManyClients(general_path, nbclients=nbclients)
    evaluation = evaluateDatasetRatio(dataset)
    storeDataset(dataset, name)
    return evaluation


# getMeanAndStd(general_path, 50)
generateAndStore('2d_dataset_mostly_2.pickle', nbclients=300)


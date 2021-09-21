import cv2
import sys
import numpy as np
import os
from tensorflow.keras.layers import Flatten
import pickle
import random
import re
import pydicom as dicom

masks_path = "NSCLC2 - Lung_Cancers3/manifest-1603198545583/NSCLC-Radiomics/LUNG1-002/masks"
images_path = "NSCLC2 - Lung_Cancers3/manifest-1603198545583/NSCLC-Radiomics/LUNG1-002/images"
# dcm_path = "NSCLC2 - Lung_Cancers3/manifest-1603198545583/NSCLC-Radiomics/LUNG1-001/09-18-2008-StudyID-NA-69331/0.000000-NA-82046"
dcm_path = "NSCLC2 - Lung_Cancers3/manifest-1603198545583/NSCLC-Radiomics/LUNG1-002/01-01-2014-StudyID-NA-85095/1.000000-NA-61228"
general_path = "NSCLC2 - Lung_Cancers3/manifest-1603198545583/NSCLC-Radiomics"


def displayImage(image):
    cv2.imshow('Window name', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def isYellow(pixel):
    ''' Return True if the pixel of the mask is yellow (if it is in the segmented zone) '''
    if pixel[0] == 36:
        return True


def crop(image_3d, z, y, x): #vertical, horizontal
    ''' Return a 32x32 image with top left corner of coordonate(y, x) '''
    crop_img = image_3d[z:z + 32, y:y + 32, x:x + 32]
    return crop_img


def isFullYellow(mask_3d, z, y, x):
    ''' Return True if the 4 corners of the 32x32 image are yellow '''
    if isYellow(mask_3d[z, y, x]) and isYellow(mask_3d[z+32, y, x]) and isYellow(mask_3d[z, y+32, x]) and isYellow(mask_3d[z, y, x+32]) and isYellow(mask_3d[z+32, y+32, x]) and isYellow(mask_3d[z+32, y, x+32]) and isYellow(mask_3d[z, y+32, x+32]) and isYellow(mask_3d[z+32, y+32, x+32]):
        return True


def allFullYellow(mask_3d, slicesnbr, jump=20):
    ''' Return a list of coordonates of the image that are full yellow
        args:
            mask: the image with the segmented tumor
            jump: the jump in the range'''
    allcoords = []
    for z in range(0, slicesnbr-32, jump):
        for y in range(0, 480, jump):
            for x in range(0, 480, jump):
                if (isFullYellow(mask_3d, z, y, x)):
                    allcoords.append((z, y, x))
    return allcoords


def isFullPurple(mask_3d, z, y, x):
    ''' Return True if the 4 corners of the 32x32 image are yellow '''
    if not (isYellow(mask_3d[z, y, x]) or isYellow(mask_3d[z+32, y, x]) or isYellow(mask_3d[z, y+32, x]) or isYellow(mask_3d[z, y, x+32]) or isYellow(mask_3d[z+32, y+32, x]) or isYellow(mask_3d[z+32, y, x+32]) or isYellow(mask_3d[z, y+32, x+32]) or isYellow(mask_3d[z+32, y+32, x+32])):
        return True


def allFullPurple(mask_3d, slicesnbr, jump=200):
    ''' Return a list of coordonates of the image that are full purple
        args:
            mask: the image with the segmented tumor
            jump: the jump in the range'''
    allcoords = []
    for z in range(0, slicesnbr-32, jump):
        for y in range(0, 480, jump):
            for x in range(0, 480, jump):
                if (isFullPurple(mask_3d, z, y, x)):
                    allcoords.append((z, y, x))
    return allcoords


def randomFullPurple(mask, slicesnbr, nbr = 20):
    ''' Generate at most nbr random coordonates of full purple coordonates
        args:
            mask: the 512 x 512 mask of the segmentation
            nbr: the number of random pixel generated (kept only if full purple)
        returns:
            purples, the list of full purple pixels coordonates'''
    purples = []
    for i in range(nbr):
        z = random.randint(0, slicesnbr - 33)
        y = random.randint(0, 479)
        x = random.randint(0, 479)
        if isFullPurple(mask, z, y, x):
            purples.append((z, y, x))
    return purples


def prepareAllYellows(allyellows, image_3d, slice_thickness):
    ''' Prepare the data for all tumor subimages from one 2d image
        args:
            allyellows: list of top left corner pixels from image to select
            image: the 2d image
        returns:
            data: the list of subimages + label to add to the dataset'''
    images_list = []
    for pixel in allyellows:
        flatten_subimage = crop(image_3d, pixel[0], pixel[1], pixel[2]).flatten()
        flatten_subimage = np.append(flatten_subimage, pixel[0] * slice_thickness)
        flatten_subimage = np.append(flatten_subimage, pixel[1])
        flatten_subimage = np.append(flatten_subimage, pixel[2])
        images_list.append(flatten_subimage)
    labels_list = np.ones((len(allyellows),), dtype=int)
    data = list(zip(images_list, labels_list))
    return data


def prepareAllPurples(allpurples, image_3d, slice_thickness):
    ''' Prepare the data for all non-tumor subimages from one 2d image
        args:
            allyellows: list of top left corner pixels from image to select
            image: the 2d image
        returns:
            data: the list of subimages + label to add to the dataset'''
    images_list = []
    for pixel in allpurples:
        flatten_subimage = crop(image_3d, pixel[0], pixel[1], pixel[2]).flatten()
        flatten_subimage = np.append(flatten_subimage, pixel[0] * slice_thickness)
        flatten_subimage = np.append(flatten_subimage, pixel[1])
        flatten_subimage = np.append(flatten_subimage, pixel[2])
        images_list.append(flatten_subimage)
    labels_list = np.zeros((len(allpurples),), dtype=int)
    data = list(zip(images_list, labels_list))
    return data


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


def getSliceThickness(dcm_path):
    slices = [dicom.dcmread(dcm_path + '/' + s) for s in os.listdir(dcm_path)[1:]]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))  # Sort by z axis
    print(dcm_path)
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    return slice_thickness


def generateDatasetFromOneClient(masks_path, images_path, dcm_path):
    ''' Generate the full 2d dataset from one client
        args:
            masks_path: directory path to all the masks of the client
            images_path: directory path to all images of the client
        returns:
            dataset: the full dataset of the client'''
    image_3d = []
    mask_3d = []
    dataset = []
    masks_files = sorted_alphanumeric(os.listdir(masks_path))
    print(masks_files)
    slice_thickness = getSliceThickness(dcm_path)
    slicesnbr = len(masks_files)
    print(slice_thickness)
    for i in range(len(masks_files)):
        mask_file = masks_path + "/mask_" + str(i) + ".png"
        mask = cv2.imread(mask_file)
        mask_3d.append(mask)

        image_file = images_path + "/image_" + str(i) + ".png"
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        image = image/255
        image_3d.append(image)

    mask_3d = np.array(mask_3d)
    image_3d = np.array(image_3d)
    allyellows = allFullYellow(mask_3d, slicesnbr)
    allpurples = randomFullPurple(mask_3d, slicesnbr)
    print("yellows: " + str(len(allyellows)))
    print("purples: " + str(len(allpurples)))
    dataset.extend(prepareAllYellows(allyellows, image_3d, slice_thickness))
    dataset.extend(prepareAllPurples(allpurples, image_3d, slice_thickness))
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
    for f in files[1:nbclients]:
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


generateAndStore('small_3d_dataset.pickle', 10)






# crop(image, fullyellows[200][0], fullyellows[200][1])

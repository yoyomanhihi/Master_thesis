import pickle
import re
import numpy as np
import pydicom as dicom
import sys
import cv2
import os
import random
import time
import seaborn

#yellow = 215, purple = 30
general_path = "NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics"
storing_path = "NSCLC-Radiomics/manifest-1603198545583"

MEAN = -741.7384087183515
STD = 432.83608694943786


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



def storeDataset(dataset, dir):
    """ Store the dataset as a pickle file"""
    with open(dir, 'wb') as output:
        pickle.dump(dataset, output)



def generateDatasetFromOneClient(masks_path, arrays_path):
    inputs = []
    outputs = []
    masks_files = sorted_alphanumeric(os.listdir(masks_path))
    for i in range(len(masks_files)):
        mask_file = masks_path + "/mask_" + str(i) + ".png"
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if np.sum(mask) > 8000000:  # 7864320: # If there is a tumor #CHECK
            if random.randint(0,3) == 2:
                print((i, np.sum(mask)))
                mask[mask < 40] = 0 # Set out of tumor to 0
                mask[mask > 210] = 1 # Set out of tumor to 1
                # print(np.sum(mask) / (512*512)) # Get zone/background ratio
                array_file = arrays_path + "/array_" + str(i) + ".npy"
                image = np.load(array_file)
                image = image - MEAN
                image = image / STD
                inputs.append(image)
                outputs.append(mask)
    data = list(zip(inputs, outputs))
    return data


def generateDatasetFromManyClients(storing_path, organ, nbclients):
    arrays_path = storing_path + "/arrays"
    masks_path = storing_path + "/masks_" + organ
    dataset = []
    files = os.listdir(masks_path)
    files.sort()
    size = min(nbclients, len(files) - 2)
    for patient in files[:size]:
        masks_path2 = masks_path + "/" + patient
        print(masks_path2)
        arrays_path2 = arrays_path + "/" + patient
        print(arrays_path2)
        dataset.extend(generateDatasetFromOneClient(masks_path2, arrays_path2))
    return dataset


def generateAndStore(name, organ, nbclients):
    ''' Generate a dataset from many clients and store it in the files
        args:
            name: name of the file to save
            nbclients: number of clients to be considered to create the dataset
        retu
            evalutation: tuple of the form (count0, count1)
            count0: number of non tumor examples
            count1: number of tumor examples'''
    dataset = generateDatasetFromManyClients(storing_path, organ, nbclients)
    dir = "datasets/" + str(name)
    storeDataset(dataset, dir)


# generateAndStore("heart_first50_1of3_geq8000000.pickle", "heart", 50)
import pickle
import re
import numpy as np
import pydicom as dicom
import cv2
import os
import matplotlib.pyplot as plt
import imageio

general_path = "NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics"
storing_path = "NSCLC-Radiomics/manifest-1603198545583"

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


def get_min_mask_number(masks_path):
    min_mask_index = 10000
    for mask in os.listdir(masks_path):
        mask_number = mask.split('_')[1]
        mask_number = int(mask_number.split('.')[0])
        if mask_number < min_mask_index:
            min_mask_index = mask_number
    return min_mask_index


def get_masks_bound(min_mask_index, percentage, nbrmasks, nbrimages):
    out_of_bounds = int(percentage*nbrmasks)
    mini = max(0, int(min_mask_index-out_of_bounds))
    maxi = min(nbrimages, int(min_mask_index+nbrmasks+out_of_bounds))
    return mini, maxi


def generateDatasetFromOneClient(masks_path, images_path, count, organ, train, percentage):
    inputs = []
    outputs = []
    min_mask_index = get_min_mask_number(masks_path)
    nbrmasks = len(os.listdir(masks_path))
    nbrimages = len(os.listdir(images_path))
    mini, maxi = get_masks_bound(min_mask_index, percentage, nbrmasks, nbrimages)
    for i in range(mini, maxi):
        mask_file = masks_path + "/mask_" + str(i) + ".png"
        image_file = images_path + "/image_" + str(i) + ".png"
        if os.path.isfile(mask_file):
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((512, 512))
        image = imageio.imread(image_file)
        plt.imsave('datasets/dataset_' + str(organ) + '/' + str(train) + f'/masks/{count}_{i}.png',
                    mask, cmap='gray')
        imageio.imwrite('datasets/dataset_' + str(organ) + '/' + str(train) + f'/images/{count}_{i}.png',
                    image.astype(np.uint16))
    data = list(zip(inputs, outputs))
    return data


def generateDatasetFromManyClients(storing_path, organ, train, initclient, endclient, initialcount, percentage):
    images_path = storing_path + "/images"
    masks_path = storing_path + "/masks_" + organ
    dataset = []
    files = os.listdir(masks_path)
    print(len(files))
    files.sort()
    end = min(endclient, len(files))
    print('end: ' + str(end))
    count = initialcount
    for patient in files[initclient:end]:
        masks_path2 = masks_path + "/" + patient
        print(masks_path2)
        images_path2 = images_path + "/" + patient
        print(images_path2)
        dataset.extend(generateDatasetFromOneClient(masks_path2, images_path2, count, organ, train, percentage))
        count += 1
    return dataset


def generateAndStore(path, organ, train, initclient, endclient, initialcount, percentage):
    ''' Generate a dataset from many clients and store it in the files
        args:
            name: name of the file to save
            nbclients: number of clients to be considered to create the dataset
        retu
            evalutation: tuple of the form (count0, count1)
            count0: number of non tumor examples
            count1: number of tumor examples'''
    generateDatasetFromManyClients(path, organ, train, initclient, endclient, initialcount, percentage)


# generateAndStore("manifest-1557326747206", "heart", "test", initclient=53, endclient=60, initialcount=48, percentage=0.1)
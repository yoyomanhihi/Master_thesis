import pickle
import re
import numpy as np
import pydicom as dicom
import cv2
import os
import matplotlib.pyplot as plt
import imageio
import shutil

general_path = "NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics"
storing_path = "NSCLC-Radiomics/manifest-1603198545583"

def sorted_alphanumeric(data):
    """ Sort the data alphanumerically. allows to have mask_2 before mask_10 for example """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


def getZ(dcm_path):
    """ Get the slice_thickness and the initial position of z in order to compute the z position
    Args:
        dcm_path: path to the dcm files
    Return:
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
    """ Store the dataset as a pickle file """
    with open(dir, 'wb') as output:
        pickle.dump(dataset, output)


def get_min_mask_index(masks_path):
    """ Get the mask with the smaller index among those in masks_path
    Args:
        masks_path: path to the masks
    Return:
        min_mask_index: the index of the mask with the smallest index
    """
    min_mask_index = 10000 # An arbitrary large number
    for mask in os.listdir(masks_path):
        mask_number = mask.split('_')[1]
        mask_number = int(mask_number.split('.')[0])
        if mask_number < min_mask_index:
            min_mask_index = mask_number
    return min_mask_index


def get_masks_bound(min_mask_index, frac, nbrmasks, nbrimages):
    """ Get the min and max index of the masks to be used in the dataset.
        a fraction frac of the number of slices including the organ
        will be used before and after them.
    Args:
        min_mask_index: the index of the mask with the smallest index
        frac: the fraction of the dataset to be used
        nbrmasks: the number of masks in the dataset
        nbrimages: the total number of images available of the patient
    Return:
        min_mask_index: the index of the mask with the smallest index
        max_mask_index: the index of the mask with the largest index
    """
    out_of_bounds = int(frac*nbrmasks)
    mini = max(0, int(min_mask_index-out_of_bounds))
    maxi = min(nbrimages, int(min_mask_index+nbrmasks+out_of_bounds))
    print(out_of_bounds, mini, maxi)
    return mini, maxi


def generateDatasetFromOneClient(masks_path, images_path, count, organ, train, frac):
    """ Generate the dataset from one client
    Args:
        masks_path: path to the masks
        images_path: path to the images
        count: the number of the client
        organ: the organ to be used
        train: if the dataset is for train, validation or test
        frac: the fraction of images to be used before and after the ones with the organ
    """

    # Get the min and max index of the masks to be used in the dataset
    min_mask_index = get_min_mask_index(masks_path)
    nbrmasks = len(os.listdir(masks_path))
    nbrimages = len(os.listdir(images_path))
    mini, maxi = get_masks_bound(min_mask_index, frac, nbrmasks, nbrimages)

    # for all images in the range mini to maxi
    for i in range(mini, maxi):
        mask_file = masks_path + "/mask_" + str(i) + ".png"
        image_file = images_path + "/image_" + str(i) + ".png"
        # if the mask exists
        if os.path.isfile(mask_file):
            # read the mask
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        # if the mask does not exist
        else:
            # create a black mask
            mask = np.zeros((512, 512))
        # read the image
        image = imageio.imread(image_file)

        # save the image and the mask
        plt.imsave('datasets/dataset_' + str(organ) + '/' + str(train) + f'/masks/{count}_{i}.png',
                    mask, cmap='gray')
        imageio.imwrite('datasets/dataset_' + str(organ) + '/' + str(train) + f'/images/{count}_{i}.png',
                    image.astype(np.uint16))


def generateDatasetFromManyClients(storing_path, organ, train, initclient, endclient, initialcount, frac):
    """ Generate the dataset from many clients
    Args:
        storing_path: path to the datasetµ
        organ: the organ to be used
        train: if the dataset is for train, validation or test
        initclient: the first client to include in the dataset
        endclient: the last client to include in the dataset
        initialcount: the index of the first client to include in the dataset
        frac: the fraction of images to be used before and after the ones with the organ
    """

    # create the path to images and masks
    images_path = storing_path + "/images"
    masks_path = storing_path + "/masks_" + organ

    # sort the clients
    files = os.listdir(masks_path)
    files.sort()

    # define the index of the last client to add in the dataset
    end = min(endclient, len(files))
    print('end: ' + str(end))

    # start the counter used as an index
    count = initialcount

    # for all clients in the range initclient to end
    for patient in files[initclient:end]:

        #create the path to the images and masks of the patient
        masks_path2 = masks_path + "/" + patient
        print(masks_path2)
        images_path2 = images_path + "/" + patient
        print(images_path2)

        # generate the dataset from the patient
        generateDatasetFromOneClient(masks_path2, images_path2, count, organ, train, frac)
        # increment the counter
        count += 1


def copyFromCentralToFederated(central_path, federated_path, client_nbr, train, initclient, endclient, remove=False):
    """ Copy the dataset from the central to folder to the federated folder
    Args:
        central_path: path to the central dataset
        federated_path: path to the federated dataset
        client_nbr: the number of the client in the federated dataset
        train: if the dataset is for train, validation or test
        initclient: the first client to include in the dataset
        endclient: the last client to include in the dataset
        remove: if the images should be removed from the central dataset
    """

    centr = central_path + '/' + train
    masks_centr = centr + '/masks'
    images_centr = centr + '/images'
    fed = federated_path + '/' + str(client_nbr) + '/' + train
    masks_fed = fed + '/masks'
    images_fed = fed + '/images'

    # for all masks in the range initclient to end
    for mask in os.listdir(masks_centr):
        # get the index of the client
        client = int(mask.split('_')[0])

        # if the client is in the range initclient to end
        if client >= initclient and client < endclient:
            src_mask = masks_centr + '/' + mask
            dst_mask = masks_fed + '/' + mask
            # move or copy the mask from the central dataset to the federated dataset
            if remove:
                shutil.move(src_mask, dst_mask)
            else:
                shutil.copy(src_mask, dst_mask)
            src_image = images_centr + '/' + mask
            dst_image = images_fed + '/' + mask
            # move or copy the image from the central dataset to the federated dataset
            if remove:
                shutil.move(src_image, dst_image)
            else:
                shutil.copy(src_image, dst_image)


def copyNonEmptyOnly(initial_path, new_path):
    """ Copy the images and masks with non-empty masks of the organ from
        initial_path to new_path
    Args:
        initial_path: path of the initial dataset
        new_path: path to the new dataset
    """

    masks_init = initial_path + '/masks'
    images_init = initial_path + '/images'
    masks_new = new_path + '/masks'
    images_new = new_path + '/images'

    # for all masks in the initial dataset
    for mask_ref in os.listdir(masks_init):
        # read the mask
        src_mask = masks_init + '/' + mask_ref
        mask = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)
        # if the mask is not empty
        if np.max(mask) > 0:
            dst_mask = masks_new + '/' + mask_ref
            shutil.copy(src_mask, dst_mask)
            src_image = images_init + '/' + mask_ref
            dst_image = images_new + '/' + mask_ref
            # copy the image from the initial dataset to the new dataset
            shutil.copy(src_image, dst_image)


def copyNonEmptyOnly2(initial_path, new_path):
    for file1 in os.listdir(initial_path):
        src_file1 = initial_path + '/' + file1
        dst_file1 = new_path + '/' + file1
        if not os.path.exists(dst_file1):
            os.makedirs(dst_file1)
        for file2 in os.listdir(src_file1):
            src_file = src_file1 + '/' + file2
            mask = cv2.imread(src_file, cv2.IMREAD_GRAYSCALE)
            if np.max(mask) > 0:
                dst_file = dst_file1 + '/' + file2
                shutil.copy(src_file, dst_file)
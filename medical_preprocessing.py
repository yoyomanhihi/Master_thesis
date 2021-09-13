import cv2
import sys
import numpy as np
import os
from tensorflow.keras.layers import Flatten
import pickle
import random

np.set_printoptions(threshold=sys.maxsize)

masks_path = "NSCLC2 - Lung_Cancers3/manifest-1603198545583/NSCLC-Radiomics/LUNG1-001/masks"
images_path = "NSCLC2 - Lung_Cancers3/manifest-1603198545583/NSCLC-Radiomics/LUNG1-001/images"
general_path = "NSCLC2 - Lung_Cancers3/manifest-1603198545583/NSCLC-Radiomics"

# cv2.imshow('Window name', image)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def displayImage(image):
    cv2.imshow('Window name', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def isYellow(pixel):
    ''' Return True if the pixel of the mask is yellow (if it is in the segmented zone) '''
    if pixel[0] == 36:
        return True


def crop(image, y, x): #vertical, horizontal
    ''' Return a 32x32 image with top left corner of coordonate(y, x) '''
    crop_img = image[y:y + 32, x:x + 32]
    return crop_img


def isFullYellow(mask, y, x):
    ''' Return True if the 4 corners of the 32x32 image are yellow '''
    if isYellow(mask[y, x]) and isYellow(mask[y + 32, x]) and isYellow(mask[y, x + 32]) and isYellow(mask[y + 32, x + 32]):
        return True


def allFullYellow(mask, jump=30):
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


def randomFullPurple(mask):
    purples = []
    for i in range(5):
        y = random.randint(0, 479)
        x = random.randint(0, 479)
        if isFullPurple(mask, y, x):
            purples.append((y, x))
    return purples


def prepareAllYellows(allyellows, image):
    ''' Prepare the data for all tumor subimages from one 2d image
        args:
            allyellows: list of top left corner pixels from image to select
            image: the 2d image
        returns:
            data: the list of subimages + label to add to the dataset'''
    images_list = []
    for pixel in allyellows:
        flatten_subimage = crop(image, pixel[0], pixel[1]).flatten()
        flatten_subimage = np.append(flatten_subimage, pixel[0])
        flatten_subimage = np.append(flatten_subimage, pixel[1])
        images_list.append(flatten_subimage)
    labels_list = np.ones((len(allyellows),), dtype=int)
    data = list(zip(images_list, labels_list))
    return data


def prepareAllPurples(allpurples, image):
    ''' Prepare the data for all non-tumor subimages from one 2d image
        args:
            allyellows: list of top left corner pixels from image to select
            image: the 2d image
        returns:
            data: the list of subimages + label to add to the dataset'''
    images_list = []
    for pixel in allpurples:
        flatten_subimage = crop(image, pixel[0], pixel[1]).flatten()
        flatten_subimage = np.append(flatten_subimage, pixel[0])
        flatten_subimage = np.append(flatten_subimage, pixel[1])
        images_list.append(flatten_subimage)
    labels_list = np.zeros((len(allpurples),), dtype=int)
    data = list(zip(images_list, labels_list))
    return data


def generateDatasetFromOneClient(masks_path, images_path):
    ''' Generate the full 2d dataset from one client
        args:
            masks_path: directory path to all the masks of the client
            images_path: directory path to all images of the client
        returns:
            dataset: the full dataset of the client'''
    dataset = []
    masks_files = os.listdir(masks_path)
    for i in range(len(masks_files)):
        mask_file = masks_path + "/mask_" + str(i) + ".png"
        mask = cv2.imread(mask_file)
        image_file = images_path + "/image_" + str(i) + ".png"
        image = cv2.imread(image_file)
        allyellows = allFullYellow(mask)
        allpurples = randomFullPurple(mask)
        dataset.extend(prepareAllYellows(allyellows, image))
        dataset.extend(prepareAllPurples(allpurples, image))
    return dataset


def evaluateDatasetRatio(dataset):
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


def storeDataset(dataset):
    with open('my_dataset.pickle', 'wb') as output:
        pickle.dump(dataset, output)


def generateDatasetForManyClients(general_path, nbclients = 300):
    dataset = []
    files = os.listdir(general_path)
    files.sort()
    for f in files[2:nbclients]:
        newpath = general_path + "/" + f
        images_path = newpath + "/images"
        masks_path = newpath + "/masks"
        dataset.extend(generateDatasetFromOneClient(masks_path, images_path))
    return dataset


def generateAndStore():
    dataset = generateDatasetForManyClients(general_path)
    evaluation = evaluateDatasetRatio(dataset)
    storeDataset(dataset)
    return evaluation

generateAndStore()




# crop(image, fullyellows[200][0], fullyellows[200][1])

import cv2
import sys
import numpy as np
import os

np.set_printoptions(threshold=sys.maxsize)

masks_path = "NSCLC2 - Lung_Cancers3/manifest-1603198545583/NSCLC-Radiomics/LUNG1-001/masks"
images_path = "NSCLC2 - Lung_Cancers3/manifest-1603198545583/NSCLC-Radiomics/LUNG1-001/images"

# cv2.imshow('Window name', image)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

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


def allFullPurple(mask, jump=40):
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


def prepareAllYellows(allyellows, image):
    ''' Prepare the data for all tumor subimages from one 2d image
        args:
            allyellows: list of top left corner pixels from image to select
            image: the 2d image
        returns:
            data: the list of subimages + label to add to the dataset'''
    images_list = []
    for pixel in allyellows:
        images_list.append(crop(image, pixel[0], pixel[1]))
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
        images_list.append(crop(image, pixel[0], pixel[1]))
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
        allpurples = allFullPurple(mask)
        if len(allyellows) > 0:
            dataset.append(prepareAllYellows(allyellows, image))
        dataset.extend(prepareAllPurples(allpurples, image))
        return dataset





generateDatasetFromOneClient(masks_path, images_path)

# crop(image, fullyellows[200][0], fullyellows[200][1])

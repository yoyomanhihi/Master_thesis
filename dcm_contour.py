from pathlib import Path
from pydicom.errors import InvalidDicomError
import dicom_contour.contour as dcm_contour
from scipy.sparse import csc_matrix
import pydicom as dicom
import numpy as np
import warnings
import matplotlib.pyplot as plt
from PIL import Image
import operator
import os
import cv2
import shutil
import scipy.misc

image_path = "NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics/LUNG1-001/09-18-2008-StudyID-NA-69331/0.000000-NA-82046"
contour_path = 'NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics/LUNG1-001/09-18-2008-StudyID-NA-69331/0.000000-NA-82046/1-1.dcm'
general_path = "NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics"

contour_data = dicom.read_file(contour_path)

print(dcm_contour.get_roi_names(contour_data))


def get_contour_file(path):
    """
    Get contour file from a given path by searching for ROIContourSequence
    inside dicom data structure.
    More information on ROIContourSequence available here:
    http://dicom.nema.org/medical/dicom/2016c/output/chtml/part03/sect_C.8.8.6.html

    Inputs:
            path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
    Return:
        contour_file (str): name of the file with the contour
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    # get .dcm contour file
    fpaths = [path + f for f in os.listdir(path) if '.dcm' in f]
    n = 0
    contour_file = None
    for fpath in fpaths:
        f = dicom.read_file(fpath)
        if 'ROIContourSequence' in dir(f):
            contour_file = fpath.split('/')[-1]
            n += 1
    if n > 1: warnings.warn("There are multiple contour files, returning the last one!")
    if contour_file is None: print("No contour file found in directory")
    return contour_file


def parse_dicom_file(filename):
    """Parse the given DICOM filename
    :param filename: filepath to the DICOM file to parse
    :return: dictionary with DICOM image data
    """
    try:
        dcm = dicom.read_file(filename)
        dcm_image = dcm.pixel_array

        try:
            intercept = dcm.RescaleIntercept
        except AttributeError:
            intercept = 0.0
        try:
            slope = dcm.RescaleSlope
        except AttributeError:
            slope = 0.0

        if intercept != 0.0 and slope != 0.0:
            dcm_image = dcm_image*slope + intercept
        return dcm_image
    except InvalidDicomError:
        return None


def coord2pixels(contour_dataset, path):
    """
    Given a contour dataset (a DICOM class) and path that has .dcm files of
    corresponding images. This function will return img_arr and contour_arr (2d image and contour pixels)
    Inputs
        contour_dataset: DICOM dataset class that is identified as (3006, 0016)  Contour Image Sequence
        path: string that tells the path of all DICOM images
    Return
        img_arr: 2d np.array of image with pixel intensities
        contour_arr: 2d np.array of contour with 0 and 1 labels
    """

    contour_coord = contour_dataset.ContourData
    # x, y, z coordinates of the contour in mm
    coord = []
    for i in range(0, len(contour_coord), 3):
        coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))

    # extract the image id corresponding to given countour
    # read that dicom file
    img_ID = contour_dataset.ContourImageSequence[0].ReferencedSOPInstanceUID
    dcm_path = path + img_ID + '.dcm' #CHANGED
    img = dicom.read_file(dcm_path)
    img_arr = parse_dicom_file(dcm_path)

    # physical distance between the center of each pixel
    x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])
    print(x_spacing, y_spacing)

    # this is the center of the upper left voxel
    origin_x, origin_y, _ = img.ImagePositionPatient

    # y, x is how it's mapped
    pixel_coords = [(np.ceil((y - origin_y) / y_spacing), np.ceil((x - origin_x) / x_spacing)) for x, y, _ in coord]

    # get contour data for the image
    rows = []
    cols = []
    for i, j in list(set(pixel_coords)):
        rows.append(i)
        cols.append(j)
    contour_arr = csc_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int8, shape=(img_arr.shape[0], img_arr.shape[1])).toarray()

    return img_arr, contour_arr, img_ID




def cfile2pixels(file, path, ROIContourSeq=0):
    """
    Given a contour file and path of related images return pixel arrays for contours
    and their corresponding images.
    args:
        file: filename of contour
        path: path that has contour and image files
        ROIContourSeq: tells which sequence of contouring to use default 0
    return:
        img_contour_arrays: A list which have pairs of img_arr and contour_arr for a given contour file
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    f = dicom.read_file(path + file)
    # index 0 means that we are getting RTV information
    GTV = f.ROIContourSequence[ROIContourSeq]
    # get contour datasets in a list
    contours = [contour for contour in GTV.ContourSequence]
    img_contour_arrays = [coord2pixels(cdata, path) for cdata in contours]
    return img_contour_arrays



def get_contour_dict(contour_file, path, index):
    """
    Returns a dictionary as k: img fname, v: [corresponding img_arr, corresponding contour_arr]
    Inputs:
        contour_file: .dcm contour file name
        path: path which has contour and image files
    Returns:
        contour_dict: dictionary with 2d np.arrays
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    # img_arr, contour_arr, img_fname
    contour_list = cfile2pixels(contour_file, path, index)

    contour_dict = {}
    for img_arr, contour_arr, img_id in contour_list:
        contour_dict[img_id] = [img_arr, contour_arr]

    return contour_dict



# get all image-contour array pairs
contour_arrays = cfile2pixels(file="1-1.dcm", path=image_path, ROIContourSeq=0)

# get first image - contour array
first_image, first_contour, img_id = contour_arrays[4]

# # show an example
# plt.figure(figsize=(20, 10))
# plt.subplot(1,2,1)
# plt.imshow(first_image)
# plt.subplot(1,2,2)
# plt.imshow(first_contour)
#
# plt.show()


def slice_order(path):
    """
    Takes path of directory that has the DICOM images and returns
    a ordered list that has ordered filenames
    Inputs
        path: path that has .dcm images
    Returns
        ordered_slices: ordered tuples of filename and z-position
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    slices = []
    for s in os.listdir(path):
        try:
            f = dicom.read_file(path + '/' + s)
            f.pixel_array  # to ensure not to read contour file
            slices.append(f)
        except:
            continue

    slice_dict = {s.SOPInstanceUID: s.ImagePositionPatient[-1] for s in slices}
    ordered_slices = sorted(slice_dict.items(), key=operator.itemgetter(1))
    return ordered_slices


# ordered files
ordered_slices = slice_order(image_path)

def get_data(path, index):
    """
    Generate image array and contour array
    Inputs:
        path (str): path of the the directory that has DICOM files in it
        contour_dict (dict): dictionary created by get_contour_dict
        index (int): index of the desired ROISequence
    Returns:
        images and contours np.arrays
    """
    images = []
    contours = []
    # handle `/` missing
    if path[-1] != '/': path += '/'
    # get contour file
    contour_file = get_contour_file(path)
    # get slice orders
    ordered_slices = slice_order(path)
    # get contour dict
    contour_dict = get_contour_dict(contour_file, path, index)

    for k,v in ordered_slices:
        # get data from contour dict
        if k in contour_dict:
            images.append(contour_dict[k][0])
            contours.append(contour_dict[k][1])
        # get data from dicom.read_file
        else:
            dcm_path = path + k + '.dcm'
            img_arr = parse_dicom_file(dcm_path)
            contour_arr = np.zeros_like(img_arr)
            images.append(img_arr)
            contours.append(contour_arr)

    return np.array(images), np.array(contours)


# images, contours = get_data(image_path, index=0)
#
#
# for img_arr, contour_arr in zip(images[79:80], contours[79:80]):
#     dcm_contour.plot2dcontour(img_arr, contour_arr)
#
# cntr = contours[80]
# plt.imshow(cntr)
#
# plt.show()

def get_index(dcm_path, index_name):
    """ Return the index number corresponding to the index name, in the ROI sequence of the patient
        args:
            dcm_path: path to the dcm_files
            index_name: name of the index to find in the ROI sequence of the patient
        return:
            i: number of the index corresponding to the index name
        """
    contour_path = dcm_path + "/1-1.dcm"
    contour_data = dicom.read_file(contour_path)
    roi_names = dcm_contour.get_roi_names(contour_data)
    for i in range(len(roi_names)):
        if roi_names[i] == index_name:
            return i
    print('index named {} not found in roi sequence: {}'.format(index_name, roi_names))
    print(dcm_path)



def create_image_mask_files(path, index_name, img_format='png'):
    """
    Create image and corresponding mask files under to folders '/images' and '/masks'
    in the parent directory of path.
    Inputs:
        path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
        index (int): index of the desired ROISequence
        img_format (str): image format to save by, png by default
    """
    # Extract Arrays from DICOM
    index = get_index(path, index_name)
    X, Y = get_data(path, index)
    for slice in X:
        slice[0][0] = 3100
        slice[511, 511] = -1025
    # X = np.clip(X, -600, 400)
    Y = np.array([dcm_contour.fill_contour(y) if y.max() == 1 else y for y in Y])
    # Create images and masks folders
    new_path = '/'.join(path.split('/')[:-2])
    images_dir = new_path + '/images/'
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir)
    masks_dir = new_path + '/masks/'
    if os.path.exists(masks_dir):
        shutil.rmtree(masks_dir)
    os.makedirs(masks_dir)
    for i in range(len(X)):
        plt.imsave(new_path + f'/images/image_{i}.{img_format}', X[i, :, :])
        plt.imsave(new_path + f'/masks/mask_{i}.{img_format}', Y[i, :, :])


def create_image_mask_forall(general_path, index_name):
    """ Create images and masks folders for every patient
        args:
            general_path: path to the folder including all patients
            index_name: name of the index to be segmented in the masks folder
    """
    patients_folders = os.listdir(general_path)
    for folder in patients_folders:
        newpath = general_path + "/" + folder
        if(os.path.isdir(newpath)):
            newfiles = os.listdir(newpath)
            for f2 in newfiles:
                if f2 != 'images' and f2 != 'masks':
                    newpath2 = newpath + "/" + f2
                    newfiles2 = os.listdir(newpath2)
                    for f3 in newfiles2:
                        newpath3 = newpath2 + "/" + f3
                        newfiles3 = os.listdir(newpath3)
                        if len(newfiles3) > 5:
                            create_image_mask_files(newpath3, index_name, img_format='png')


# def print_shapes(general_path):
#     patients_folders = os.listdir(general_path)
#     for folder in patients_folders:
#         newpath = general_path + "/" + folder
#         if(os.path.isdir(newpath)):
#             newpath2 = newpath + "/" + "images"
#             newfiles2 = os.listdir(newpath2)
#             print(newfiles2)
#             newpath3 = newpath2 + "/" + newfiles2[0]
#             im = cv2.imread(newpath3)
#             print(im.shape)

# create_image_mask_forall(general_path, 'GTV-1')



from pathlib import Path
import dicom_contour.contour as dcm
import pydicom as dicom
import numpy as np
import matplotlib.pyplot as plt
import operator
import os
import cv2

image_path = "NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics/LUNG1-128/08-04-2006-NA-NA-03538/0.000000-NA-06742"
contour_path = 'NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics/LUNG1-128/08-04-2006-NA-NA-03538/0.000000-NA-06742/1-1.dcm'
general_path = "NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics"

contour_data = dicom.read_file(contour_path)

print(dcm.get_roi_names(contour_data))


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
    img_contour_arrays = [dcm.coord2pixels(cdata, path) for cdata in contours]
    return img_contour_arrays

# get all image-contour array pairs
contour_arrays = cfile2pixels(file="1-1.dcm", path=image_path, ROIContourSeq=1)

# get first image - contour array
first_image, first_contour, img_id = contour_arrays[2]

# show an example
plt.figure(figsize=(20, 10))
plt.subplot(1,2,1)
plt.imshow(first_image)
plt.subplot(1,2,2)
plt.imshow(first_contour)

plt.show()


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
    contour_file = dcm.get_contour_file(path)
    # get slice orders
    ordered_slices = slice_order(path)
    # get contour dict
    contour_dict = dcm.get_contour_dict(contour_file, path, index)

    for k,v in ordered_slices:
        # get data from contour dict
        if k in contour_dict:
            images.append(contour_dict[k][0])
            contours.append(contour_dict[k][1])
        # get data from dicom.read_file
        else:
            img_arr = dicom.read_file(path + k + '.dcm').pixel_array
            contour_arr = np.zeros_like(img_arr)
            images.append(img_arr)
            contours.append(contour_arr)

    return np.array(images), np.array(contours)


# images, contours = get_data(image_path, index=0)
#
#
# for img_arr, contour_arr in zip(images[79:80], contours[79:80]):
#     dcm.plot2dcontour(img_arr, contour_arr)

# cntr = contours[80]
# plt.imshow(cntr)
#
# plt.show()

def get_index(path, index_name):
    contour_path = path + "/1-1.dcm"
    contour_data = dicom.read_file(contour_path)
    roi_names = dcm.get_roi_names(contour_data)
    for i in range(len(roi_names)):
        if roi_names[i] == index_name:
            return i
    print('index named {} not found in roi sequence: {}'.format(index_name, roi_names))
    print(path)


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
    Y = np.array([dcm.fill_contour(y) if y.max() == 1 else y for y in Y])

    # Create images and masks folders
    new_path = '/'.join(path.split('/')[:-2])
    os.makedirs(new_path + '/images/', exist_ok=True)
    os.makedirs(new_path + '/masks/', exist_ok=True)
    for i in range(len(X)):
        plt.imsave(new_path + f'/images/image_{i}.{img_format}', X[i, :, :])
        plt.imsave(new_path + f'/masks/mask_{i}.{img_format}', Y[i, :, :])


def create_image_mask_forall(general_path, index_name):
    patients_folders = os.listdir(general_path)

    for folder in patients_folders[251:]:
        newpath = general_path + "/" + folder
        if(os.path.isdir(newpath)):
            newfiles = os.listdir(newpath)
            for f2 in newfiles:
                newpath2 = newpath + "/" + f2
                newfiles2 = os.listdir(newpath2)
                for f3 in newfiles2:
                    newpath3 = newpath2 + "/" + f3
                    newfiles3 = os.listdir(newpath3)
                    if len(newfiles3) > 5:
                        create_image_mask_files(newpath3, index_name, img_format='png')


def print_shapes(general_path):
    patients_folders = os.listdir(general_path)

    for folder in patients_folders:
        newpath = general_path + "/" + folder
        if(os.path.isdir(newpath)):
            newpath2 = newpath + "/" + "images"
            newfiles2 = os.listdir(newpath2)
            print(newfiles2)
            newpath3 = newpath2 + "/" + newfiles2[0]
            im = cv2.imread(newpath3)
            print(im.shape)

# create_image_mask_forall(general_path, 'GTV-1') # 128 first made



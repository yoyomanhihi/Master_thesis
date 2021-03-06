"""
This file is largely based on the library "dicom_contour" by KeremTurgutlu
Many functions were however modified or added to fit the needs of the project.
https://github.com/KeremTurgutlu/dicom-contour
"""

from pydicom.errors import InvalidDicomError
import dicom_contour.contour as dcm_contour
from scipy.sparse import csc_matrix
import pydicom as dicom
import numpy as np
import warnings
import matplotlib.pyplot as plt
import operator
import os
import cv2
import shutil
from PIL import Image, ImageDraw
import imageio

def plot2dcontour(img_arr, contour_arr, img_nbr, figsize=(20, 20)):
    """ Show 2d MR img with contour
    Args:
        img_arr: 2d np.array image array with pixel intensities
        contour_arr: 2d np.array contour array with pixels of 1 and 0
    """

    masked_contour_arr = []
    for i in range(len(contour_arr)):
        masked_contour_arr.append(np.ma.masked_where(contour_arr[i] == 0, contour_arr[i]))
    # plt.figure(figsize=figsize)
    # plt.subplot(1, 2, 1)
    # title = "tumor zone image " + str(img_nbr)
    # plt.title(title, fontsize=18)
    plt.imshow(img_arr, cmap='gray', interpolation='none')
    # plt.subplot(1, 2, 2)
    # plt.imshow(img_arr, cmap='gray', interpolation='none')
    for i in range(len(masked_contour_arr)):
        plt.imshow(masked_contour_arr[i], cmap='cool', interpolation='none', alpha=0.7)
    plt.show()


def get_contour_file(path):
    """ Get contour file from a given path by searching for ROIContourSequence
        inside dicom data structure.
        More information on ROIContourSequence available here:
        http://dicom.nema.org/medical/dicom/2016c/output/chtml/part03/sect_C.8.8.6.html
    Args:
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
    """ Parse the given DICOM filename
    Args:
        filename: filepath to the DICOM file to parse
    Return:
        dictionary with DICOM image data
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

        # set outside scanner to air
        outside_image = np.min(dcm_image)
        dcm_image[dcm_image == outside_image] = -1000

        return dcm_image
    except InvalidDicomError:
        return None


def coord2pixels(contour_dataset, path):
    """ Given a contour dataset (a DICOM class) and path that has .dcm files of
        corresponding images. This function will return img_arr and contour_arr (2d image and contour pixels)
    Inputs:
        contour_dataset: DICOM dataset class that is identified as (3006, 0016)  Contour Image Sequence
        path: string that tells the path of all DICOM images
    Return:
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
    # print(x_spacing, y_spacing)

    # this is the center of the upper left voxel
    origin_x, origin_y, _ = img.ImagePositionPatient

    # y, x is how it's mapped
    pixel_coords = [(np.ceil((y - origin_y) / y_spacing), np.ceil((x - origin_x) / x_spacing)) for x, y, _ in coord]

    pixel_coords2 = []

    for i in range(len(pixel_coords)):
        pixel_coords2.append((pixel_coords[i][1], pixel_coords[i][0]))

    # show image
    # img = Image.new('L', (512, 512), 0)
    # ImageDraw.Draw(img).polygon(pixel_coords2, outline=1, fill=0)
    # mask = np.array(img)
    # plt.imshow(mask, cmap='gray')
    # plt.show()

    img = Image.new('L', (512, 512), 0)
    ImageDraw.Draw(img).polygon(pixel_coords2, outline=1, fill=0) # CHECK
    mask = np.array(img)

    # get contour data for the image
    rows = []
    cols = []
    for i, j in list(set(pixel_coords)):
        rows.append(i)
        cols.append(j)
    contour_arr = csc_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int8, shape=(img_arr.shape[0], img_arr.shape[1])).toarray()

    return img_arr, contour_arr, img_ID, mask




def cfile2pixels(file, path, ROIContourSeq=0):
    """ Given a contour file and path of related images return pixel arrays for contours
        and their corresponding images.
    Args:
        file: filename of contour
        path: path that has contour and image files
        ROIContourSeq: tells which sequence of contouring to use default 0
    Return:
        img_contour_arrays: A list which have pairs of img_arr and contour_arr for a given contour file
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    f = dicom.read_file(path + file)

    # moda = f.Modality
    # print(moda)

    # print(dcm_contour.get_roi_names(f))

    GTV = f.ROIContourSequence[ROIContourSeq]
    # get contour datasets in a list
    contours = [contour for contour in GTV.ContourSequence]
    print(np.shape(contours))
    img_contour_arrays = [coord2pixels(cdata, path) for cdata in contours]

    # show image 2
    # allcontours = []
    # for array in img_contour_arrays:
    #     if len(allcontours) == 0:
    #         allcontours = array[3]
    #         plt.imshow(allcontours, cmap='gray')
    #     else:
    #         allcontours += array[3]
    # plt.imshow(allcontours, cmap='gray')
    # plt.show()

    return img_contour_arrays


def get_contour_dict(contour_file, path, index):
    """ Return a dictionary as k: img fname, v: [corresponding img_arr, corresponding contour_arr]
    Args:
        contour_file: .dcm contour file name
        path: path which has contour and image files
    Return:
        contour_dict: dictionary with 2d np.arrays
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    # img_arr, contour_arr, img_fname
    contour_list = cfile2pixels(contour_file, path, index)

    contour_dict = {}
    for img_arr, contour_arr, img_id, mask in contour_list:
        if img_id not in contour_dict.keys(): # CHANGED
            contour_dict[img_id] = [img_arr, contour_arr, mask]
        else:
            contour_dict[img_id].extend([img_arr, contour_arr, mask])

    # show contours
    keys = list(contour_dict.keys())
    for i in range(len(keys)):
        allcontours = contour_dict[list(contour_dict.keys())[i]]
        toshow = []
        print(len(allcontours))
        if len(allcontours) > 3:
            for j in range(1, len(allcontours), 3):
                if j == 1:
                    toshow = allcontours[j]
                else:
                    toshow += allcontours[j]
            # plt.imshow(toshow, cmap='gray')
            # plt.show()
            plt.imsave('pics/right/' + str(list(contour_dict.keys())[i]) + '.png', toshow, cmap='gray')


    return contour_dict

def slice_order(path):
    """ Takes path of directory that has the DICOM images and returns
        a ordered list that has ordered filenames
    Args:
        path: path that has .dcm images
    Return:
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


def get_data(path, index):
    """ Generate image array and contour array
    Args:
        path (str): path of the the directory that has DICOM files in it
        contour_dict (dict): dictionary created by get_contour_dict
        index (int): index of the desired ROISequence
    Return:
        images and contours np.arrays
    """
    images = []
    contours = []
    masks = []
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
            for i in range(0, len(contour_dict[k]), 3):
                if i == 0:
                    images.append(contour_dict[k][i])
                    contours.append([contour_dict[k][i+1]])
                    masks.append([contour_dict[k][i+2]])
                if i > 0:
                    contours[len(contours)-1] = np.append(contours[len(contours)-1], [contour_dict[k][i+1]], axis=0)
                    masks[len(masks)-1] = np.append(masks[len(masks)-1], [contour_dict[k][i+2]], axis=0)
        # get data from dicom.read_file
        else:
            dcm_path = path + k + '.dcm'
            img_arr = parse_dicom_file(dcm_path)
            contour_arr = np.zeros_like(img_arr)
            images.append(img_arr)
            contours.append([contour_arr])
            masks.append([contour_arr])


    return np.array(images), np.array(contours), np.array(masks)


def get_index(dcm_path, index_name):
    """ Return the index number corresponding to the index name, in the ROI sequence of the patient
        Args:
            dcm_path: path to the dcm_files
            index_name: name of the index to find in the ROI sequence of the patient
        Return:
            i: number of the index corresponding to the index name
        """
    contour_path = dcm_path + "/1-1.dcm"
    contour_data = dicom.read_file(contour_path)
    roi_names = dcm_contour.get_roi_names(contour_data)
    print(roi_names)
    for i in range(len(roi_names)):
        if roi_names[i] == index_name:
            return i
    print('index named {} not found in roi sequence: {}'.format(index_name, roi_names))
    print(dcm_path)


def create_images_files(path, img_format='png'):
    """ Create image and corresponding mask files under to folders '/images' and '/masks'
        in the parent directory of path.
    Args:
        path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
        index (int): index of the desired ROISequence
        img_format (str): image format to save by, png by default
    """
    print(path)
    index = get_index(path, "Heart")
    images, contours, masks = get_data(path, index) #Make sure to read the file as the index doesn't matter

    patient = path.split('/')[-2]
    new_path = '/'.join(path.split('/')[:-4]) # Check
    images_dir = new_path + '/images/' + patient
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir)
    for i in range(len(images)):
        image = images[i]
        image[image<-1000] = -1000
        image[0][0] = 3200
        image = (65535 * (image + 1000) / image.ptp()).astype(np.uint16)
        imageio.imwrite(images_dir + f'/image_{i}.{img_format}', image.astype(np.uint16))
        print(images_dir + f'/image_{i}.{img_format}')



def create_images_forall(general_path):
    """ Create images and masks folders for every patient
    Args:
        general_path: path to the folder including all patients
        index_name: name of the index to be segmented in the masks folder
    """
    patients_folders = os.listdir(general_path)
    i = 0 # CHECK
    for folder in patients_folders[i:]:
        print(i)
        newpath = general_path + "/" + folder
        if(os.path.isdir(newpath)):
            newfiles = os.listdir(newpath)
            for f2 in newfiles:
                if f2 != 'images' and f2 != 'masks' and f2 != 'masks_Lung_Left' and f2 != 'arrays' and f2 != 'masks_Lungs' and f2 != 'masks_Lung_Right':
                    newpath2 = newpath + "/" + f2
                    newfiles2 = os.listdir(newpath2)
                    for f3 in newfiles2:
                        newpath3 = newpath2 + "/" + f3
                        newfiles3 = os.listdir(newpath3)
                        if len(newfiles3) > 5:
                            create_images_files(newpath3, img_format='png')
                            i+=1




def create_masks_files_only(path, index_name, img_format='png'):
    """ Create image and corresponding mask files under to folders '/images' and '/masks'
        in the parent directory of path.
    Args:
        path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
        index (int): index of the desired ROISequence
        img_format (str): image format to save by, png by default
    """
    # Extract Arrays from DICOM
    print(path)
    index = get_index(path, index_name)
    if index is not None:
        images, contours, masks = get_data(path, index)
        Y = []
        for mask in masks:
            if len(mask[0]) == 1:
                Y.append(mask[0])
            else:
                ctr = mask[0]
                for i in range(1, len(mask), 1):
                    ctr += mask[i]
                ctr[ctr>1] = 0
                Y.append(ctr)
        Y = np.array(Y)
        # Create images and masks folders
        patient = path.split('/')[-2] # CHECK
        new_path = '/'.join(path.split('/')[:-4])
        masks_dir = new_path + '/masks_Lung_Right/' + patient #CHECK
        if os.path.exists(masks_dir):
            shutil.rmtree(masks_dir)
        os.makedirs(masks_dir)
        for i in range(len(images)):
            mask = Y[i, :, :]
            if np.sum(mask) > 0:
                print("save path: " + masks_dir + f'/mask_{i}.{img_format}')
                # plt.imsave(masks_dir + f'/mask_{i}.{img_format}', Y[i, :, :], cmap="gray")



def create_masks_only_forall(general_path, index_name):
    """
    Create images and masks folders for every patient
    Args:
        general_path: path to the folder including all patients
        index_name: name of the index to be segmented in the masks folder
    """
    patients_folders = os.listdir(general_path)
    i = 0  # CHECK
    for folder in patients_folders[i:]:
        print(i)
        newpath = general_path + "/" + folder
        if(os.path.isdir(newpath)):
            newfiles = os.listdir(newpath)
            for f2 in newfiles:
                if f2 != 'images' and f2 != 'masks' and f2 != 'masks_Lung_Left' and f2 != 'masks_Lung_Right' and f2 != "masks_Lungs" and f2 != 'arrays':
                    newpath2 = newpath + "/" + f2
                    newfiles2 = os.listdir(newpath2)
                    for f3 in newfiles2:
                        newpath3 = newpath2 + "/" + f3
                        newfiles3 = os.listdir(newpath3)
                        if len(newfiles3) > 5:
                            create_masks_files_only(newpath3, index_name, img_format='png')
                            i+=1


# Pre:
'''
storing_path   
???
????????????masks_Lung_Right
???   ???
???   ????????????patient1
???   ???    ???   image1.png
???   ???    ???   image2.png
???   ???    ???   ...
???   ???    
???   ????????????patient2
???   ???    ???   image1.png
???   ???    ???   image2.png
???   ???    ???   ...
???   ???    
????????????masks_Lung_Left
    ???
    ????????????patient1
    ???    ???   image1.png
    ???    ???   image2.png
    ???    ???   ...
    ???    
    ????????????patient2
         ???   image1.png
         ???   image2.png
         ???   ...
'''
def merge_masks_lungs_forall(storing_path):
    """ Merge masks_Lung_Right and masks_Lung_Left folders into
        masks_Lungs folder.
        All images of the same patient with the same name are
        merged into one image with the two lungs.
    Args:
        storing_path (str): path to the folder described in Pre
    """

    path_left = storing_path + "/masks_Lung_Left"
    path_right = storing_path + "/masks_Lung_Right"

    # for all patients
    for i in range(len(os.listdir(path_left))):
        patient = os.listdir(path_left)[i]
        path_client_left = path_left + "/" + patient
        path_client_right = path_right + "/" + patient
        dir = storing_path + '/masks_lung/' + patient
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
        # for all masks of the left lung
        for mask_ref in (os.listdir(path_client_left)):
            number_ref_1 = mask_ref.split("_")[1]
            number_ref = int(number_ref_1.split(".")[0])
            newpath_left = path_client_left + "/" + mask_ref
            newpath_right = path_client_right + "/" + mask_ref
            # if there is a corresponding right lung, merge them
            if(os.path.exists(newpath_right)):
                mask_left = cv2.imread(newpath_left, cv2.IMREAD_GRAYSCALE)
                mask_right = cv2.imread(newpath_right, cv2.IMREAD_GRAYSCALE)
                mask_lungs = mask_left + mask_right
                plt.imsave(dir + f'/mask_{number_ref}.png', mask_lungs, cmap='gray')
            # else juste save the left part
            else:
                mask_left = cv2.imread(newpath_left, cv2.IMREAD_GRAYSCALE)
                plt.imsave(dir + f'/mask_{number_ref}.png', mask_left, cmap='gray')
        # for all right lungs
        for mask_ref in (os.listdir(path_client_right)):
            number_ref_1 = mask_ref.split("_")[1]
            number_ref = int(number_ref_1.split(".")[0])
            newpath_left = path_client_left + "/" + mask_ref
            newpath_right = path_client_right + "/" + mask_ref
            # save right lung if left didn't exist
            if not os.path.exists(newpath_left):
                mask_right = cv2.imread(newpath_right, cv2.IMREAD_GRAYSCALE)
                plt.imsave(dir + f'/mask_{number_ref}.png', mask_right, cmap='gray')


def merge_contours():
    """ Merge contours of the same patient into one image.
        Useful for the report.
    """
    pass
    left = 'pics/left/'
    right = 'pics/right/'
    for file in os.listdir(left):
        if os.path.exists(right + file):
            right_cont = cv2.imread(right + file, cv2.IMREAD_GRAYSCALE)
            left_cont = cv2.imread(left + file, cv2.IMREAD_GRAYSCALE)
            merge = right_cont + left_cont
            plt.imsave('pics/merge/' + file + '.png', merge, cmap='gray')

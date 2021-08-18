import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

import pydicom
import scipy.ndimage

import glob

from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.morphology import disk, opening, closing

from IPython.display import HTML
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from os import listdir, mkdir


def load_scan(path):
    ''' Load the scans in given folder path and save its thickness (pixel size of the z direction)
        return:
            slices: An array of dicom files readed, corresponding to all images of the scan
        args:
            path: filepath to the data. Must be a folder of all the dicom files of the scan
    '''
    slices = [pydicom.dcmread(path + '/' + s) for s in listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2])) #Sort by z axis
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def set_outside_scanner_to_air(raw_pixelarrays):
    ''' Remove out of the image pixels by replacing them to air
        return:
            raw_pixelarrays: the pixels updated
        args:
            raw_pixelarrays: the raw pixels of the slice
    '''
    # in OSIC we find outside-scanner-regions with raw-values of -2000.
    # Let's threshold between air (0) and this default (-2000) using -1000
    outside_image = raw_pixelarrays.min()
    raw_pixelarrays[raw_pixelarrays == outside_image] = 0
    return raw_pixelarrays


def transform_to_hu(slices):
    ''' Convert the scans to HU units
        return:
            The scan updated
        args:
            slices: the array of dicom files readed
    '''

    images = np.stack([file.pixel_array for file in slices])
    images = images.astype(np.int16)

    images = set_outside_scanner_to_air(images)

    # convert to HU
    for n in range(len(slices)):

        intercept = slices[n].RescaleIntercept
        slope = slices[n].RescaleSlope

        if slope != 1:
            images[n] = slope * images[n].astype(np.float64)
            images[n] = images[n].astype(np.int16)

        images[n] += np.int16(intercept)

    return np.array(images, dtype=np.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):
    ''' Resample the scan to a new_spacing resolution
        return:
            image: The new pixels after resampling
            new_spacing: The new spacing after resampling
        args:
            image: The pixels of the 3D image of the patient
            scan: The full scan of the patient
            new_spacing: The new spacing of the pixels
    '''

    # Determine current pixel spacing
    print(scan[0].SliceThickness)
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)
    print(spacing)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def plot_3d(image, threshold=700, color="navy"):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces, _, _ = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.2)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1
    image[image < 0] = 0
    return image


def zero_center(image):
    image = image - PIXEL_MEAN
    return image


def set_manual_window(hu_image, custom_center, custom_width):
    w_image = hu_image.copy()
    min_value = custom_center - (custom_width/2)
    max_value = custom_center + (custom_width/2)
    w_image[w_image < min_value] = min_value
    w_image[w_image > max_value] = max_value
    return w_image


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        biggest = vals[np.argmax(counts)]
    else:
        biggest = None

    return biggest


def fill_lungs(binary_image):
    image = binary_image.copy()
    # For every slice we determine the largest solid structure
    for i, axial_slice in enumerate(image):
        axial_slice = axial_slice - 1
        labeling = measure.label(axial_slice)
        l_max = largest_label_volume(labeling, bg=0)

        if l_max is not None: #This slice contains some lung
            image[i][labeling != l_max] = 1
    return image


def segment_lung_mask(image, fill_lung_structures = False):
    segmented = np.zeros(image.shape)

    #Separate lungs(-700)/air(-1000) and tissues (0)
    for n in range(image.shape[0]):
        binary_image = np.array(image[n] > -320, dtype=np.int8) + 1
        labels = measure.label(binary_image)

        bad_labels = np.unique([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
        for bad_label in bad_labels:
            binary_image[labels == bad_label] = 2

        # We have a lot of remaining small signals outside of the lungs that need to be removed.
        # In our competition closing is superior to fill_lungs
        selem = disk(2)
        binary_image = opening(binary_image, selem)

        binary_image -= 1  # Make the image actual binary
        binary_image = 1 - binary_image  # Invert it, lungs are now 1

        segmented[n] = binary_image.copy() * image[n]

    return segmented


def segment_lung_mask2(image, fill_lung_structures=True):


    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image


def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def zero_center(image):
    image = image - PIXEL_MEAN
    return image


# Some constants
# INPUT_FOLDER = 'OrganisedLung2 - LCTSC'
INPUT_FOLDER = 'OrganisedLung - NSCLS-Radomics-Interobserver1'
patients = listdir(INPUT_FOLDER)
patients.sort()
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25

allscans = list()
for file in listdir(INPUT_FOLDER):
    allscans.append(load_scan(INPUT_FOLDER + '/' + file))

for i in range(len(allscans)):

    first_patient = allscans[i]

    hu_scans = transform_to_hu(first_patient)

    pix_resampled, spacing = resample(hu_scans, first_patient, [1,1,1])

    plot_3d(pix_resampled, 400)

    segmented_lungs = segment_lung_mask(pix_resampled)
    segmented_lungs2 = segment_lung_mask2(pix_resampled, False)
    # segmented_lungs_fill = segment_lung_mask2(pix_resampled, True)

    plot_3d(segmented_lungs, threshold=-600)
    # kernel = np.ones((5, 5), np.uint8)
    # dilated_one = cv2.dilate(segmented_lungs2, kernel, iterations=1)
    plot_3d(segmented_lungs2, 0)
    # plot_3d_full(segmented_lungs, threshold=-600)
    # plot_3d_full(segmented_lungs2, 0)
    # plot_3d(segmented_lungs_fill, 0, color="green")

    plt.show()
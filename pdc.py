import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from sklearn.cluster import KMeans
from skimage import morphology

import pydicom
import scipy.ndimage
from rt_utils import RTStructBuilder

import pandas as pd
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.morphology import disk, opening, closing

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

        print(slices[n].PatientID)
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


def segment_lung_mask_open(image):
    segmented = np.zeros(image.shape)

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


def segment_lung_mask_closed(image):
    segmented = np.zeros(image.shape)

    #Separate lungs(-700)/air(-1000) and tissues (0)
    for n in range(image.shape[0]):
        binary_image = np.array(image[n] > -320, dtype=np.int8) + 1
        labels = measure.label(binary_image)

        bad_labels = np.unique([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])

        for bad_label in bad_labels:
            binary_image[labels == bad_label] = 2

        binary_image_2 = binary_image.copy()
        for bad_label in bad_labels:
            binary_image_2[labels == bad_label] = 2

        selem = disk(2)
        closed_binary_2 = closing(binary_image_2, selem)

        closed_binary_2 -= 1  # Make the image actual binary
        closed_binary_2 = 1 - closed_binary_2  # Invert it, lungs are now 1

        segmented[n] = closed_binary_2.copy() * image[n]


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


# Standardize the pixel values
def make_lungmask(img, display=False):
    row_size = img.shape[0]
    col_size = img.shape[1]

    # mean = np.mean(img)
    # std = np.std(img)
    # img = img - mean
    # img = img / std

    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the
    # underflow and overflow on the pixel spectrum
    img[img == max] = mean
    img[img == min] = mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
    dilation = morphology.dilation(eroded, np.ones([8, 8]))

    labels = measure.label(dilation)  # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < row_size / 10 * 9 and B[3] - B[1] < col_size / 10 * 9 and B[0] > row_size / 5 and B[
            2] < col_size / 5 * 4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size, col_size], dtype=np.int8)
    mask[:] = 0

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask
    #
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask * img, cmap='gray')
        ax[2, 1].axis('off')

        plt.show()
    return mask * img


def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 512
    desired_width = 178
    desired_height = 512
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Resize across z-axis
    img = scipy.ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


#################################### Trying the functions ###############################################


# Some constants
# INPUT_FOLDER = "NSCLC2 - Lung_Cancers3/manifest-1603198545583/NSCLC-Radiomics/LUNG1-001/09-18-2008-StudyID-NA-69331"
# INPUT_FOLDER = 'OrganisedLung2 - LCTSC'
# INPUT_FOLDER = 'OrganisedLung - NSCLS-Radomics-Interobserver1'
INPUT_FOLDER = "NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics/LUNG1-001/09-18-2008-StudyID-NA-69331/0.000000-NA-82046"
patients = listdir(INPUT_FOLDER)
patients.sort()
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25

FILE = "manifest-1638281314414/Pediatric-CT-SEG/Pediatric-CT-SEG-00DCF4D6/10-09-2009-NA-CT-45894/30144.000000-CT-67414/1-001.dcm"
# FILE = "NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics/LUNG1-001/09-18-2008-StudyID-NA-69331/0.000000-NA-82046/1.3.6.1.4.1.32722.99.99.104776423223073991273677458347669148913.dcm"
# FILE = "manifest-1622561851074/NSCLC Radiogenomics/AMC-001/04-30-1994-NA-PETCT Lung Cancer-74760/3.000000-CT FUSION-97864/1-090.dcm"

dataset = pydicom.dcmread(FILE)
print(dataset.ImagePositionPatient)
print(np.shape(dataset.pixel_array))
plt.imshow(dataset.pixel_array, cmap="gray")
plt.show()

hu = transform_to_hu([dataset])
plt.imshow(hu[0], cmap="gray")
plt.show()

# def make_all(allscans):
#     for i in range(len(allscans)):
#         first_patient = allscans[i]
#         hu_scans = transform_to_hu(first_patient)
#         segmented_lungs_closed = segment_lung_mask_closed(hu_scans)
#         plot_3d(segmented_lungs_closed, threshold=-600)
#
#
# def save_HU(allscans):
#     for i in range(len(allscans)):
#         first_patient = allscans[i]
#         hu_scans = transform_to_hu(first_patient)
#         np.save("Lung_HU/LCTSC/" + str(i), hu_scans)
#
#
# def save_segmented(allscans):
#     PREPROCESSED_FOLDER = "Lung_segmented/LCTSC"
#     for file in listdir(PREPROCESSED_FOLDER):
#         hu_scans = np.load(PREPROCESSED_FOLDER + "/" + file)
#         segmented_lungs_closed = segment_lung_mask_closed(hu_scans)
#         np.save("Lung_segmented/LCTSC/" + str(file), segmented_lungs_closed)
#
#
# def save_resized():
#     PREPROCESSED_FOLDER = "Lung_segmented/LCTSC"
#     for file in listdir(PREPROCESSED_FOLDER):
#         segmented = np.load(PREPROCESSED_FOLDER + "/" + file)
#         resized = resize_volume(segmented)
#         np.save("Lung_resized/LCTSC/" + str(file), resized)
#
#
# allscans = list()
# for file in listdir(INPUT_FOLDER):
#     allscans.append(load_scan(INPUT_FOLDER + '/' + file))
#
# make_all(allscans)

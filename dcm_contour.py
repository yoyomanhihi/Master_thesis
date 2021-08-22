from pathlib import Path
import dicom_contour.contour as dcm
import pydicom as dicom
import numpy as np

image_path = "NSCLC2 - Lung_Cancers3/manifest-1603198545583/NSCLC-Radiomics/LUNG1-001/09-18-2008-StudyID-NA-69331/0.000000-NA-82046"
contour_path = 'NSCLC2 - Lung_Cancers3/manifest-1603198545583/NSCLC-Radiomics/LUNG1-001/09-18-2008-StudyID-NA-69331/0.000000-NA-82046/1-1.dcm'

contour_data = dicom.read_file(contour_path)

print(contour_data)

print(dcm.get_roi_names(contour_data))


def cfile2pixels(file, path, ROIContourSeq=0):
    """
    Given a contour file and path of related images return pixel arrays for contours
    and their corresponding images.
    Inputs
        file: filename of contour
        path: path that has contour and image files
        ROIContourSeq: tells which sequence of contouring to use default 0 (RTV)
    Return
        contour_iamge_arrays: A list which have pairs of img_arr and contour_arr for a given contour file
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
contour_arrays = cfile2pixels(file="1-1.dcm", path=image_path, ROIContourSeq=0)
print(contour_arrays)
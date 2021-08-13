from pydicom.data import get_testdata_file
from pydicom import dcmread
from typing import List, Tuple
import matplotlib.pyplot as plt

from pydicom import dcmread
from pydicom.data import get_testdata_file
from pydicom.fileset import FileSet

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

path = "realIRM/1-080.dcm"
testpath = get_testdata_file('CT_small.dcm')
ds = dcmread(path)
print(ds)
print(ds.pixel_array)
plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
plt.show()

# Some constants
INPUT_FOLDER = 'OrganisedLung'
patients = os.listdir(INPUT_FOLDER)
patients.sort()
print(patients)


# Load the scans in given folder path
def load_scan(path):
    slices = [dcmread(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

allscans = list()
for file in os.listdir(INPUT_FOLDER):
    allscans.append(load_scan(INPUT_FOLDER + '/' + file))

print(len(allscans[21]))

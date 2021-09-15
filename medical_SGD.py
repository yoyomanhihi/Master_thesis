import pickle
import numpy as np
import medical_FL_utils_2d as med_utils_2d
import medical_FL_utils_3d as med_utils_3d

x_train, y_train, x_test, y_test = med_utils_3d.prepareTrainTest('medium_3d_dataset.pickle')
med_utils_3d.simpleSGD(x_train, y_train, x_test, y_test)

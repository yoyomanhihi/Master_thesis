import pickle
import numpy as np
import medical_FL_utils as med_utils


x_train, y_train, x_test, y_test = med_utils.prepareTrainTest('small_dataset.pickle')
med_utils.simpleSGD(x_train, y_train, x_test, y_test)

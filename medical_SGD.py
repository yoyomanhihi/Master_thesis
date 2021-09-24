import pickle
import numpy as np
import medical_FL_utils as med_utils
from tensorflow import keras
import tensorflow as tf

image_path = 'NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics/LUNG1-304/images/image_64.png'
images_path = 'NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics/LUNG1-304/images'

x_train, y_train, x_test, y_test = med_utils.prepareTrainTest_2d('small_2d_dataset_1.pickle')

model = med_utils.simpleSGD_2d(x_train, y_train, x_test, y_test)

model.save('small_2d_model.h5')
#
model = keras.models.load_model('2d_model.h5')
#
test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))
SGD_acc = med_utils.test_model(x_test, y_test, model)


med_utils.segmentation_2d(model, image_path)
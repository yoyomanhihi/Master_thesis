import pickle
import numpy as np
import medical_FL_utils as med_utils
from tensorflow import keras
import tensorflow as tf

image_path = 'NSCLC2 - Lung_Cancers3/manifest-1603198545583/NSCLC-Radiomics/LUNG1-301/images/image_68.png'
images_path = 'NSCLC2 - Lung_Cancers3/manifest-1603198545583/NSCLC-Radiomics/LUNG1-301/images'

x_train, y_train, x_test, y_test = med_utils.prepareTrainTest_3d('small_3d_dataset.pickle')

# model = med_utils.simpleSGD_3d(x_train, y_train, x_test, y_test)

# model.save('small_3d_model.h5')

model = keras.models.load_model('small_3d_model.h5')

test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))
SGD_acc = med_utils.test_model(x_test, y_test, model)

med_utils.segmentation_3d(model, images_path)
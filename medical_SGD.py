import medical_FL_utils as med_utils
import tensorflow as tf
from tensorflow import keras


def make_all():
    client_path = 'NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics/LUNG1-419'

    # x_train, y_train, x_test, y_test = med_utils.prepareTrainTest_2d('2d_dataset_cnn.pickle', "cnn")
    #
    # model = med_utils.simpleSGD_2d(x_train, y_train, x_test, y_test, strategy="cnn")
    #
    # model.save('2d_model_cnn.h5')
    #
    model1 = keras.models.load_model('2d_model_new.h5')
    model2 = keras.models.load_model('2d_model_cnn.h5')
    #
    # test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))
    # SGD_acc = med_utils.test_model(x_test, y_test, model)

    # print(SGD_acc)

    # for i in range(95, 100, 2):
    #     med_utils.segmentation_2d(model1, client_path, i, 8, strategy="dense")
    #     med_utils.segmentation_2d(model2, client_path, i, 8, strategy="cnn")

    med_utils.segmentation_2d(model1, client_path, 78, 8, strategy="dense")


    # return SGD_acc

make_all()
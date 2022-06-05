# Master thesis: Federated learning for organ segmentation

- **Author**    : Misonne Thibaud
- **Promotor**  : Jodogne Sébastien
- **Academic year** : 2021-2022

This repository contains all functions, classes and utilities needed to reproduce federated averaging and federated equal-chances for organ segmentation based on CT-scans, using the U-Net architecture. This includes dicom and rt-struct management, keras manipulations and federated learning algorithms. This README will help you understanding how it works, file by file.

# Main files

## dcm_contour.py

This file contains all functions necessary to process datasets of CT-scan images on the DICOM format. It also deals with RT-STRUCT which are the data structures used to store the hand-made segmentations, necessary for deep learning. As mentionned on the top of the file, many functions are inspired by the library "dicom_contour" by KeremTurgutlu. Most functions were however not directly working on the datasets, which is why I had to modify them internally to make in work on my datasets. The library can be found here: https://github.com/KeremTurgutlu/dicom-contour

## unet_preprocessing.py

This file contains all functions necessary to transform the raw datasets into organised datasets to be usable by U-Net.

## unet_utils.py

This is the most important of all files. Inside, you can find every function useful to train U-Net using federated learning or a standard SGD. This includes the U-Net architecture, the Dice's loss function etc.

Federated learning functions inspired by the great tutorial available here: https://towardsdatascience.com/federated-learning-a-step-by-step-implementation-in-tensorflow-aac568283399

U-Net model inspired by another great tutorial that can be found here:
https://medium.com/@fabio.sancinetti/u-net-convnet-for-ct-scan-segmentation-6cc0d465eed3

## unet_segmentation.py

This file includes functions useful to make the segmentations and display them once a model was built.

## unet_running.py

This files uses the functions defined in unet_utils.py and abstracts everything to functions directly callable to train or evaluate models.

# Secondary files

## file_utils.py

This file includes small functions to read and write into txt files. Used to plot graphics of the training scores.

## lr_scheduler.py

This file was taken from https://github.com/yui-mhcp, a very nice collaborative github including many utilities for deep learning. 

This file contains different strategies for a learning rate scheduler. In other words, it creates an object that you can pass in the train function of keras in order to slightly reduce the learning rate while the training is processing. However, this didn't result in improving results in practice.

## plots.py

This file contains functions to plot the evoluation of the score and the value of the loss functions during training. The graphs were however not very interesting and where not included in the report of the thesis anyway.

## requirements.txt

Classical requirements file including the libraries to be loaded to make the code work

## main.py

main function to build a model using federated equal-chances. It is however more an example file to show of it must be called, as the function needs a pre-processed dataset in the first time to be working, and the goal of this thesis is not to make it directly functional without any pre-processing. 
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

## unet_segmentation.py

This file includes functions useful to make the segmentations and display them once a model was built.

## unet_running.py

This files uses the functions defined in unet_utils.py and abstracts everything to functions directly callable to train or evaluate models.

# Secondary files


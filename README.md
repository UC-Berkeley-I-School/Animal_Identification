# Animal_Identification
Capstone Project for Animal Identification

## 1. Prepare Dataset and convert to Yolov5 format. 

Download leopard dataset from (https://lila.science/datasets/leopard-id-2022/). Run Animal_Identification/notebook/create_leopard_dataset.ipynb to convert the data from Coco format to Yolov5 format.   The data_path and labels_path must be set correctly to point to downloaded dataset.  Download Zebra and Giraffe datasets from https://lila.science/datasets for generating negative samples.

The downloaded Dataset is split to train and test set by create_leopard_dataset.ipynb.

## 2. EDA on Leopard Dataset

The EDA is done in Animal_Identification/notebook/Exploratory_Analysis.ipynb 


## 3. Generate Negative Samples and augmentations for Leopard Dataset.

The negatives samples are in Animal_Identification/notebook/Negative_Samples.ipynb and augmentation of Leopard dataset in im_augmentation.ipynb

## 8. Baseline Model.
The baseline model is implemented in Animal_Identification/blob/branch_july24/leopard_baseline.ipynb. The components of teh base line model - especially the vanilla triplet loss function is borrowed from https://github.com/adambielski/siamese-triplet.

## 5. Yolov5 training and fine turning.

The Pytorch implementation of Yolov5 module was downloaded from https://github.com/ultralytics/yolov5. The Yolov5 training/testing is described in Animal_Identification/yolov5/tutorial.ipynb.  We also used Roboflow for Yolov5 training and verification


## 6. Image resize and oraganize as 64 classes 

The cropped images from Yolov5 are resized to required dimensions for identification CNN. There are 64 leopard classes with 7 or more example images in the train set. So, these 64 leopards classes are identified and strored as separate folder (train and test) for traing the Siamsese Network.  This is done in Animal_Identification/resize_image_parts.ipynb.


## 7. Siamese Network Training 
The training with combined cross entropy and triplet loss is done in Animal_Identification/blob/branch_july24/leopard_identification.ipynb.   

## 8. Inference on Leopard Images
The inference pipeline can take any leopard images (in a folder) and runs Yolov5 to create cropped images.  The re-identification or new leopard detection is done on these cropped images.   The pipeline is implemented in Animal_Identification/blob/branch_july24/leopard_inference.ipynb




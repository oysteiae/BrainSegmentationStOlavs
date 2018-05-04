# BrainSegmentationStOlavs
Work in progress :construction_worker:
Sorry for the terrible commit messages.

## Acknowledgments
The CNN architecture is the same as this https://github.com/GUR9000/Deep_MRI_brain_extraction which was implemented in Theano. This implementation started out as trying to write their implementation in keras. Some functions used for processing and finding data have been copyied from that repo. These are marked. Their model is published in this paper:
Kleesiek, J., Urban, G., Hubert, A., Schwarz, D., Maier-Hein, K., Bendszus, M., Biller,A.,  2016.  Deep  MRI  brain  extraction:  A  3D  convolutional  neural  network  for  skullstripping. NeuroImage 129, 460–469

Some inspiration for the 3D U-Net have been taken from this https://github.com/ellisdg/3DUnetCNN repo. Mainly the way the final prediction is built from predicted patches.

This 3D U-Net model is a keras implementation of a model introduced in this paper:
Cicek, ̈O.,  Abdulkadir,  A.,  Lienkamp,  S.  S.,  Brox,  T.,  Ronneberger,  O.,  2016.  3D  U-net:  Learning dense volumetric segmentation from sparse annotation. Lecture Notesin Computer Science (including subseries Lecture Notes in Artificial Intelligence andLecture Notes in Bioinformatics) 9901 LNCS, 424–432.

## Prerequisites
To use this code you should have a CUDA and cuDNN enabled GPU. This implementation is tested mainly on CUDA v9.0 and cuDNN version 7.0.
The listed libraries with versions are libraries that I have tested, therefor it could work with other versions aswell.

[keras] 2.1.5
[tensorflow-gpu] 1.1.0
[numpy] 1.14.2
[nibabel] 2.1.0

## Usage
# Training
 - mode: Train or test
 - arc: cnn or unet
 - nepochs: How may epochs should the model be trained for. Note that the training will stop regardless after a certain amount of epoch due to way the training is implemented
 - savename: What the trained model will be saved as and also how the logs etc will be saved.
 - data: folder/folders containing data that the model should train on.
 - labels: folder/folders corresponding labels for the data. Note that the data and labels should correspond in their order alphabetically
 - gpus: number of gpus the model should be trained on
 - use-validation: if specified the implementation will split the data into training, test and validation data and save their indices to disk
 - training_with_slurm: bool to used for getting the implementation to save correctly when using NTNUs EPIC cluster
 - validation_data: same as data, but used for validation
 - validation_labels: same as labels, but used for validation

# Prediction

# Evaluation


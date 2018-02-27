from CNN.Predictor3DCNN import Predictor3DCNN
from CNN.Trainer3DCNN import Trainer3DCNN
from Unet.Trainer3DUnet import Trainer3DUnet
from Unet.Predictor3DUnet import Predictor3DUnet
from Callbacks.Logger import LossHistory
import helper
import ntpath

import nibabel as nib
import numpy as np
from os import listdir as _listdir, getcwd
from os.path import isfile as _isfile,join as  _join, abspath, splitext
from pathlib import Path

def normalize_all_data():
    d = helper.load_files(["D:\\MRI_SCANS\\data"])
    data = helper.process_data(d)
    j = 0
    for i in range(0, len(d)):
        d_split = d[i].split('.')
        if(d_split[-1] == "img"):
            nin = nib.Nifti1Image(data[j], None, nib.load(d[i]).header)
            nin.to_filename("D:\\MRI_SCANS\\Normalized\\data\\" + ntpath.basename(d[i]).split('.')[0] + "_processed.nii.gz")

            j += 1

def process_all_labels():
    l = helper.load_files(["D:\\MRI_SCANS\\labels"])
    labels = helper.process_labels(l)
    j = 0
    for i in range(0, len(l)):
        d_split = l[i].split('.')
        if(d_split[-1] == "img"):
            nin = nib.Nifti1Image(labels[j], None, nib.load(l[i]).header)
            nin.to_filename("D:\\MRI_SCANS\\Normalized\\labels\\" + ntpath.basename(l[i]).split('.')[0] + "_processed.nii.gz")

            j += 1

# TODO: Look at data Augmentation
# TODO: Write your own function for finding the optimal input size.
# TODO: You can add error checking
# Correct MRI scans for the "pollution" in code.
# Resampling
# Implement the loss thing.
# TODO Fix voxel size output
# TODO reshape input to same voxel size?
def main():
    #trainer = Trainer3DCNN(["D:\\MRI_SCANS\\data"], ["D:\\MRI_SCANS\\labels"], "trained_on_oasis_tested_on_lbpa40", build_CNN, using_sparse_categorical_crossentropy=False, use_cross_validation=False)
    #predictor = Predictor3DCNN("models/batchsize21", ["D:\\MRI_SCANS\\predict"], build_CNN, using_sparse_categorical_crossentropy=False)
    #predictor.predict()
    #unetTrainer = build_3DUnet((144, 144, 144, 1))
    #model = Trainer3DCNN()
    #test = model.build_model()
    #unet = Trainer3DUnet((40, 40, 40, 1))
    #unet = Trainer3DUnet((176, 208, 176, 1))
    
    #unet.train(["/localdata/OASIS/data"], ["/localdata/OASIS/data"], 1000, "unet_test", batch_size=1)
    #unet.train(["D:\\MRI_SCANS\\data"], ["D:\\MRI_SCANS\\labels"], 1000, "unet_test", batch_size=4)
    unet = Predictor3DUnet("C:\\Users\\oyste\\Documents\\Visual Studio 2015\\Projects\\SkullStripping\\BrainSegmentationStOlavs\\models\\unet_test", ["D:\\MRI_SCANS\\da"], (40, 40, 40, 1))
    unet.predict()

main()
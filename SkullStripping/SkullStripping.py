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
import argparse
from tensorflow.python.client import device_lib

def normalize_all_data():
    d = helper.load_files(["D:\\MRI_SCANS\\StOlavsResampled\\data"])
    data = helper.process_data(d)
    j = 0
    for i in range(0, len(d)):
        d_split = d[i].split('.')
        nin = nib.Nifti1Image(data[j], None, nib.load(d[i]).header)
        nin.to_filename("D:\\MRI_SCANS\\NormalizedStOlavsResampled\\data\\" + ntpath.basename(d[i]).split('.')[0] + "_processed.nii.gz")
        print("Saved " + d[i])

        j += 1

def process_all_labels():
    l = helper.load_files(["D:\\MRI_SCANS\\StOlavsResampled\\labels"])
    labels = helper.process_labels(l)
    j = 0
    for i in range(0, len(l)):
        d_split = l[i].split('.')
        nin = nib.Nifti1Image(labels[j], None, nib.load(l[i]).header)
        nin.to_filename("D:\\MRI_SCANS\\NormalizedStOlavsResampled\\labels\\" + ntpath.basename(l[i]).split('.')[0] + "_processed.nii.gz")
        print("Saved " + l[i])

        j += 1

# TODO: Look at data Augmentation
# TODO: Write your own function for finding the optimal input size.
# TODO: You can add error checking
# Correct MRI scans for the "pollution" in code.
# Resampling
# Implement the loss thing.
# TODO Fix voxel size output
# TODO reshape input to same voxel size?
# TODO maybe add processing option for the program
# TODO write code that makes a seperate folder for each experiment
def main():
    #normalize_all_data()
    #process_all_labels()
    print(device_lib.list_local_devices())

    parser = argparse.ArgumentParser(description='Module for training a model or predicting using an existing model')
    parser.add_argument('--mode', dest='mode', required=True, type=str, help='Specify if training or predicting')
    parser.add_argument('--arc', dest='arc', required=True, type=str, help='Specify which arcitecture')
    parser.add_argument('--nepochs', dest='nepochs', required=False, type=int, help='How many epochs should the model be trained')
    parser.add_argument('--savename', dest='save_name', required=True, type=str, help='Path to the corresponding labels')
    parser.add_argument('--data', dest='data', required=False, type=str, nargs='+', help='Path to the data')
    parser.add_argument('--labels', dest='labels', required=False, type=str, nargs='+', help='The save name of the model')
    parser.add_argument("--gpus", dest='gpus', required=True, type=int, default=1, help="# of GPUs to use for training")
    parser.add_argument("--use_validation", dest='use_validation', required=False, type=bool, default=False)
    parser.add_argument("--training_with_slurm", dest='training_with_slurm', required=False, type=bool, default=False)
    args = parser.parse_args()
    
    if(args.mode == 'train'):
        print("Training")
        print(args.data)
        if(args.data is None or args.labels is None):
            parser.error("Requires data and labels")
        if(args.nepochs is None):
            parser.error("You must write in how many epochs")
        elif(args.arc == 'unet'):
            unet = Trainer3DUnet((64, 64, 64, 1), args.gpus)
            unet.train(args.data, args.labels, args.nepochs, args.save_name, use_validation=args.use_validation, training_with_slurm=args.training_with_slurm)        
        elif(args.arc == 'cnn'):
            model = Trainer3DCNN(args.gpus)
            model.train(args.data, args.labels, args.nepochs, args.save_name, use_validation=args.use_validation, training_with_slurm=args.training_with_slurm)
    if(args.mode == 'test'):
        if(args.data is None):
            parser.error("Requires data to make predictions")
        elif(args.arc == 'unet'):
            unet = Predictor3DUnet(args.save_name, (64, 64, 64, 1), args.gpus)
            unet.predict(args.data)
        elif(args.arc == 'cnn'):
            predictor = Predictor3DCNN(args.save_name, args.gpus)
            predictor.predict(args.data)

main()
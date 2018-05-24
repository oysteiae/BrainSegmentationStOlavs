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
from extra import dice_coefficient_loss

def normalize_all_data():
    d = helper.load_files(["D:\\MRISCANS\\LITS\\NormalizedVolumes"])
    data = helper.process_data(d)
    j = 0
    for i in range(0, len(d)):
        d_split = d[i].split('.')
        nin = nib.Nifti1Image(data[j], None, None)
        nin.to_filename("D:\\MRISCANS\\LITS\\ResampledData\\" + ntpath.basename(d[i]).split('.')[0] + "_processed.nii.gz")
        print("Saved " + d[i])

        j += 1

def process_all_labels():
    l = helper.load_files(["D:\\MRISCANS\\LITS\\ResampledLabels"])
    labels = helper.process_labels(l)
    j = 0
    for i in range(0, len(l)):
        d_split = l[i].split('.')
        #if(d_split[-1] == "img"):
        nin = nib.Nifti1Image(labels[j], None, None)
        nin.to_filename("D:\\MRISCANS\\LITS\\NormalizedResamledLabels\\" + ntpath.basename(l[i]).split('.')[0] + "_processed.nii.gz")
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
    #print(device_lib.list_local_devices())

    parser = argparse.ArgumentParser(description='Module for training a model or predicting using an existing model')
    parser.add_argument('--mode', dest='mode', required=True, type=str, help='Specify if training or predicting')
    parser.add_argument('--arc', dest='arc', required=True, type=str, help='Specify which arcitecture')
    parser.add_argument('--nepochs', dest='nepochs', required=False, type=int, help='How many epochs should the model be trained')
    parser.add_argument('--savename', dest='save_name', required=True, type=str, help='Path to the corresponding labels')
    parser.add_argument('--data', dest='data', required=False, type=str, nargs='+', help='Path to the data')
    parser.add_argument('--labels', dest='labels', required=False, type=str, nargs='+', help='The save name of the model')
    parser.add_argument("--gpus", dest='gpus', required=True, type=int, default=1, help="# of GPUs to use for training")
    parser.add_argument("--training_with_slurm", dest='training_with_slurm', required=False, type=bool, default=False)
    parser.add_argument("--use_validation", dest='use_validation', required=False, type=bool, default=False, help="If you want to use training, validation and testing data during training set this to True. Also if you want to predict on the training, validation or testing data during predictino set this to true")
    
    parser.add_argument('--validation_data', dest='validation_data', required=False, type=str, nargs='+', help='You can specify the validation data here if you don\'t want to specify it automatically using use_validation')
    parser.add_argument('--validation_labels', dest='validation_labels', required=False, type=str, nargs='+', help='The corresponding labels for the validation data')

    parser.add_argument('--patch_size', dest='patch_size', required=False, type=int, nargs='+', help='Size of patch used for input, default to (59, 59, 59, 1) for CNN and (64, 64, 64, 1) for the U-Net')
    parser.add_argument('--loss_function', dest='loss_function', required=False, type=str, help='The loss function to use, defaults to kld')
    parser.add_argument('--part_to_test_on', dest='part_to_test_on', required=False, type=str, help='Test on the training, validation or testing part of the data if use_validation is True')

    args = parser.parse_args()

    if(args.part_to_test_on is None and args.use_validation):
        part_to_test_on = 'testing_indices'
    elif(args.use_testing_data):
        part_to_test_on = args.part_to_test_on + "_indices"
    else:
        part_to_test_on = None

    if(args.mode == 'train'):
        print("Training")
        if(args.data is None or args.labels is None):
            parser.error("You need to specify data and labels")
        if(args.nepochs is None):
            parser.error("You must write in how many epochs to train for")
        elif(args.arc == 'unet'):
            if(args.patch_size == None):
                patch_size = (64, 64, 64, 1)
            else:
                patch_size = tuple(args.patch_size)

            unet = Trainer3DUnet(patch_size, args.gpus, training_with_slurm=args.training_with_slurm, loss_function=args.loss_function)
            unet.train(args.data, args.labels, args.nepochs, args.save_name, use_validation=args.use_validation, training_with_slurm=args.training_with_slurm, validation_data=args.validation_data, validation_labels=args.validation_labels)        
        elif(args.arc == 'cnn'):
            if(args.patch_size == None):
                patch_size = (59, 59, 59, 1)
            else:
                patch_size = tuple(args.patch_size)
            model = Trainer3DCNN(args.gpus, training_with_slurm=args.training_with_slurm, loss_function=args.loss_function, cnn_input_size=patch_size)
            model.train(args.data, args.labels, args.nepochs, args.save_name, use_validation=args.use_validation, training_with_slurm=args.training_with_slurm, validation_data=args.validation_data, validation_labels=args.validation_labels)
    if(args.mode == 'test'):
        if(args.data is None):
            parser.error("--data is required to make predictions")
        elif(args.arc == 'unet'):
            if(args.patch_size == None):
                patch_size = (64, 64, 64, 1)
            else:
                patch_size = tuple(args.patch_size)
            unet = Predictor3DUnet(args.save_name, patch_size, args.gpus, loss_function=args.loss_function, use_validation=args.use_validation, part_to_test_on=part_to_test_on)
            unet.predict(args.data)
        elif(args.arc == 'cnn'):
            if(args.patch_size == None):
                patch_size = (84, 84, 84, 1)
            else:
                patch_size = tuple(args.patch_size)
            predictor = Predictor3DCNN(args.save_name, args.gpus, loss_function=args.loss_function, input_size=patch_size, use_validation=args.use_validation, part_to_test_on=part_to_test_on)
            predictor.predict(args.data)

main()
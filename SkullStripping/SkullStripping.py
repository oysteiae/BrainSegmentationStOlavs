from CNN.Predictor3DCNN import Predictor3DCNN
from CNN.Trainer3DCNN import Trainer3DCNN
import helper
from Unet.Build3DUnet import build_3DUnet
from CNN.Build3DCNN import build_3DCNN
from Callbacks.Logger import LossHistory
from pathlib import Path
import os


# TODO: Look at data Augmentation
# TODO: Find out what they mean with channels
# TODO: Maybe you should randomize weights.
# TODO: Write your own function for finding the optimal input size.
# TODO: You can add error checking
# Correct MRI scans for the "pollution" in code.
# Resampling
# Implement the loss thing.
# TODO Fix voxel size output
# TODO reshape input to same voxel size?
# TODO make config
# TODO some folders should be in the subdirectory.
def main():
    #trainer = Trainer3DCNN(["D:\\MRI_SCANS\\data"], ["D:\\MRI_SCANS\\labels"], "trained_on_oasis_tested_on_lbpa40", build_CNN, using_sparse_categorical_crossentropy=False, use_cross_validation=False)
    #predictor = Predictor3DCNN("models/batchsize21", ["D:\\MRI_SCANS\\predict"], build_CNN, using_sparse_categorical_crossentropy=False)
    #predictor.predict()
    #unetTrainer = build_3DUnet((144, 144, 144, 1))
    #model = Trainer3DCNN()
    #test = model.build_model()
    #unet = build_3DUnet((144, 144, 144, 1))

main()
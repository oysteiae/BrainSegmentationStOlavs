from CNN.Predictor3DCNN import Predictor3DCNN
from CNN.Trainer3DCNN import Trainer3DCNN
import helper
from CNN.Build3DCNN import build_3DCNN
from Unet.Trainer3DUnet import Trainer3DUnet
from Callbacks.Logger import LossHistory

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
    unet = Trainer3DUnet((176, 208, 176, 1))
    unet.train(["D:\\MRI_SCANS\\da"], ["D:\\MRI_SCANS\\la"], 1000, "unet_test", batch_size=1)

main()
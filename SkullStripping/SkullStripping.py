from CNN.Predictor3DCNN import Predictor3DCNN
from CNN.Trainer3DCNN import Trainer3DCNN
from Unet.Trainer3DUnet import Trainer3DUnet
from Unet.Predictor3DUnet import Predictor3DUnet
from Callbacks.Logger import LossHistory
import argparse
import helper

# TODO: Look at data Augmentation
# TODO: Write your own function for finding the optimal input size.
# TODO: You can add error checking
# Correct MRI scans for the "pollution" in code.
# Resampling
# Implement the loss thing.
# TODO Fix voxel size output
# TODO reshape input to same voxel size?
def main():
    parser = argparse.ArgumentParser(description='Module for training a model or predicting using an existing model')
    parser.add_argument('--mode', dest='mode', required=True, type=str, help='Specify if training or predicting')
    parser.add_argument('--arc', dest='arc', required=True, type=str, help='Specify which arcitecture')
    parser.add_argument('--savename', dest='save_name', required=True, type=str, help='Path to the corresponding labels')
    parser.add_argument('--data', dest='data', required=False, type=str, nargs='+', help='Path to the data')
    parser.add_argument('--labels', dest='labels', required=False, type=str, nargs='+', help='The save name of the model')
    args = parser.parse_args()
    print(args.mode)
    if(args.mode == 'train'):
        print("Training")
        print(args.data)
        if(args.data is None or args.labels is None):
            parser.error("Requires data and labels")
        elif(args.arc == 'unet'):
            unet = Trainer3DUnet((40, 40, 40, 1))
            unet.train(args.data, args.labels, 1000, args.save_name)        
        elif(args.arc == 'cnn'):
            model = Trainer3DCNN()
            model.train(args.data, args.labels, 1000, args.save_name)
    if(args.mode == 'test'):
        if(args.data is None):
            parser.error("Requires data and labels")
        elif(args.arc == 'unet'):
            unet = Predictor3DUnet(args.save_name, args.data, (40, 40, 40, 1))
            unet.predict()
        elif(args.arc == 'cnn'):
            predictor = Predictor3DCNN(args.save_name, args.data)
            predictor.predict()

main()
from keras.layers import Activation
from keras.engine import Input, Model

from keras.optimizers import Adam
from keras.layers.convolutional import Conv3D, MaxPooling3D
import numpy as np
from Predictor3DCNN import Predictor3DCNN
from Trainer3DCNN import Trainer3DCNN
import helper

# TODO rewrite so that you can set the parameters
# TODO maybe move to a class
def build_CNN(input_shape, pool_size=(2, 2, 2),
                  initial_learning_rate=0.00001, deconvolution=False, stride=1, using_sparse_categorical_crossentropy=False):
    inputs = Input(input_shape)
    conv1 = Conv3D(filters=16, kernel_size=(4, 4, 4), strides=stride, activation='relu', padding='valid')(inputs)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(filters=24, kernel_size=(5, 5, 5), strides=stride, activation='relu', padding='valid')(pool1)
    conv3 = Conv3D(filters=28, kernel_size=(5, 5, 5), strides=stride, activation='relu', padding='valid')(conv2)
    conv4 = Conv3D(filters=34, kernel_size=(5, 5, 5), strides=stride, activation='relu', padding='valid')(conv3)
    conv5 = Conv3D(filters=42, kernel_size=(5, 5, 5), strides=stride, activation='relu', padding='valid')(conv4)
    conv6 = Conv3D(filters=50, kernel_size=(5, 5, 5), strides=stride, activation='relu', padding='valid')(conv5)
    conv7 = Conv3D(filters=50, kernel_size=(5, 5, 5), strides=stride, activation='relu', padding='valid')(conv6)

    #TODO the first argument should really be 2, I think
    conv8 = Conv3D(filters=2, kernel_size=(1, 1, 1))(conv7)
    act = Activation('softmax')(conv8)
    model = Model(inputs=inputs, outputs=act)

    print(model.summary())
    if using_sparse_categorical_crossentropy:
        print("Using sparse categorical crossentropy as loss function")
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        # The loss function should perhaps be Kullback-Leibler divergence
        print("Using Kullback-Leibler divergence as loss function")
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss='kld', metrics=['accuracy'])
    
    return model

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
def main():
    #trainer = Trainer3DCNN(["D:\\MRI_SCANS\\data"], ["D:\\MRI_SCANS\\labels"], "trained_on_oasis_tested_on_lbpa40", build_CNN, using_sparse_categorical_crossentropy=False, use_cross_validation=False)
    predictor = Predictor3DCNN("models/batchsize21", ["D:\\MRI_SCANS\\predict"], build_CNN, using_sparse_categorical_crossentropy=False)
    predictor.predict()

main()
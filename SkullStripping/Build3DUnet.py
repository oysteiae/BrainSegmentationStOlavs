import numpy as np
import keras
from keras.engine import Input, Model
from keras.layers import Activation, BatchNormalization
from keras.layers.convolutional import Conv3D, MaxPooling3D, Deconvolution3D, UpSampling3D, Conv3DTranspose
from keras.optimizers import Adam
from extra import dice_coefficient_loss
from Generator import get_generator

# 19069955 parameters
# 19,068,993
# For some reason you have less parameters.
def build_3DUnet(input_shape, use_upsampling=False, initial_learning_rate=0.00005, stride=1, kernel_size=3):
    inputs = Input(input_shape)
        
    # Don't know kernel size
    # They use batch normalization in every layer except the last in the article.
    # Layers with maxpool
    conv1 = Conv3D(filters=32, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(inputs)
    conv2 = Conv3D(filters=64, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(conv1)
    maxpool1 = MaxPooling3D(pool_size=2,)(conv2)

    conv3 = Conv3D(filters=64, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(maxpool1)
    conv4 = Conv3D(filters=128, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(conv3)
    maxpool2 = MaxPooling3D(pool_size=2)(conv4)

    conv5 = Conv3D(filters=128, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(maxpool2)
    conv6 = Conv3D(filters=256, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(conv5)
    maxpool3 = MaxPooling3D(pool_size=2)(conv6)

    #Layers with upsampling
    # You can either use deconvolution or upsampling. It seems that the code from github uses deconvolution.
    conv7 = Conv3D(filters=256, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(maxpool3)
    conv8 = Conv3D(filters=512, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(conv7)
    upsampling1 = get_upconvolution(512, use_upsampling)(conv8)
    concat1 = keras.layers.concatenate([upsampling1 ,conv6])

    conv9 = Conv3D(filters=256, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(concat1)
    conv10 = Conv3D(filters=256, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(conv9)
    upsampling2 = get_upconvolution(256, use_upsampling)(conv10)
    concat2 = keras.layers.concatenate([upsampling2 ,conv4])

    conv11 = Conv3D(filters=128, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(concat2)
    conv12 = Conv3D(filters=128, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(conv11)
    upsampling3 = get_upconvolution(128, use_upsampling)(conv12)
    concat3 = keras.layers.concatenate([upsampling3 ,conv2])

    # Final layers
    conv13 = Conv3D(filters=64, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(concat3)
    conv14 = Conv3D(filters=64, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(conv13)

    # TODO: is kernel size 1 here?
    conv15 = Conv3D(filters=1, kernel_size=1, strides=stride, activation='relu', padding='same')(conv14)
    act = Activation('softmax')(conv15)
    model = Model(inputs=inputs, outputs=act)
    # Remember to use different loss function.
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss='kld', metrics=['accuracy'])

    print(model.summary())
    return model

def get_upconvolution(nfilters, use_upsampling, kernel_size=2):
    if(use_upsampling):
        return UpSampling3D(size=kernel_size)
    else:
        return Conv3DTranspose(filters=nfilters, kernel_size=kernel_size, strides=2)

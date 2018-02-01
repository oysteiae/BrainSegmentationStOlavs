import numpy as np
import keras
from keras.engine import Input, Model
from keras.layers import Activation, BatchNormalization
from keras.layers.convolutional import Conv3D, MaxPooling3D, Deconvolution3D, UpSampling3D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from extra import dice_coefficient_loss

class Trainer3DUnet:
    """description of class"""
    def __init__(self, input_shape):
        model = self.build_3DUnet(input_shape)

    # 19069955 parameters
    def build_3DUnet(self, input_shape):
        initial_learning_rate = 0.00005
        use_upsampling = False

        inputs = Input(input_shape)
        stride = 1
        # Don't know kernel size
        # They use batch normalization in every layer except the last in the article.
        # Layers with maxpool
        conv1 = Conv3D(filters=32, kernel_size=3, strides=stride, activation='relu', padding='same')(inputs)
        conv2 = Conv3D(filters=64, kernel_size=3, strides=stride, activation='relu', padding='same')(conv1)
        maxpool1 = MaxPooling3D(pool_size=2,)(conv2)

        conv3 = Conv3D(filters=64, kernel_size=3, strides=stride, activation='relu', padding='same')(maxpool1)
        conv4 = Conv3D(filters=128, kernel_size=3, strides=stride, activation='relu', padding='same')(conv3)
        maxpool2 = MaxPooling3D(pool_size=2)(conv4)

        conv5 = Conv3D(filters=128, kernel_size=3, strides=stride, activation='relu', padding='same')(maxpool2)
        conv6 = Conv3D(filters=256, kernel_size=3, strides=stride, activation='relu', padding='same')(conv5)
        maxpool3 = MaxPooling3D(pool_size=2)(conv6)

        #Layers with upsampling
        # You can either use deconvolution or upsampling. It seems that the code from github uses deconvolution.
        conv7 = Conv3D(filters=256, kernel_size=3, strides=stride, activation='relu', padding='same')(maxpool3)
        conv8 = Conv3D(filters=512, kernel_size=3, strides=stride, activation='relu', padding='same')(conv7)
        if(use_upsampling):
            upsampling1 = UpSampling3D(size=2)(conv8)
        else:
            upsampling1 = Deconvolution3D(filters=512, kernel_size=2, strides=2)
        concat1 = keras.layers.concatenate([upsampling1 ,conv6], axis=1)

        conv9 = Conv3D(filters=256, kernel_size=3, strides=stride, activation='relu', padding='same')(concat1)
        conv10 = Conv3D(filters=256, kernel_size=3, strides=stride, activation='relu', padding='same')(conv9)
        #upsampling2 = UpSampling3D(size=2)(conv10)
        if(use_upsampling):
            upsampling2 = UpSampling3D(size=2)(conv10)
        else:
            upsampling2 = Deconvolution3D(filters=256, kernel_size=2, stride=2)(conv10)
        concat2 = keras.layers.concatenate([upsampling2 ,conv4])

        conv11 = Conv3D(filters=128, kernel_size=3, strides=stride, activation='relu', padding='same')(concat2)
        conv12 = Conv3D(filters=128, kernel_size=3, strides=stride, activation='relu', padding='same')(conv11)
        #upsampling3 = UpSampling3D(size=2)(conv12)
        if(use_upsampling):
            upsampling3 = UpSampling3D(size=2)(conv12)
        else:
            upsampling3 = Deconvolution3D(filters=128, kernel_size=2, stride=2)(conv12)
        concat3 = keras.layers.concatenate([upsampling3 ,conv2])

        # Final layers
        conv13 = Conv3D(filters=64, kernel_size=3, strides=stride, activation='relu', padding='same')(concat3)
        conv14 = Conv3D(filters=64, kernel_size=3, strides=stride, activation='relu', padding='same')(conv13)

        conv15 = Conv3D(filters=1, kernel_size=1, strides=stride, activation='relu', padding='same')(conv14)
        act = Activation('softmax')(conv15)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss='kld', metrics=['accuracy'])

        print(model.summary())

        return model


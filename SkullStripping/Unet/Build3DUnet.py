import numpy as np
import keras
from keras.engine import Input, Model
from keras.layers import Activation, BatchNormalization
from keras.layers.convolutional import Conv3D, MaxPooling3D, Deconvolution3D, UpSampling3D, Conv3DTranspose
from keras.optimizers import Adam, SGD
from extra import dice_coefficient_loss

# Apperently the network trains just fine when the input size is smaller.
# 19069955 parameters
# 19,068,993
# For some reason you have less parameters.
# Det kan hende du må synke learning rate mens du lærer her også.
def build_3DUnet(input_shape, use_upsampling=False, initial_learning_rate=0.0005, stride=1, kernel_size=3):
    padding = 'same'
    activation = 'sigmoid'
    # 8 Works too.
    n_base_filters = 32
    uses_batch_normalization = True
    use_upsampling = False
    inputs = Input(input_shape)
    print(input_shape)
    # Don't know kernel size
    # They use batch normalization in every layer except the last in the
    # article.
    # Layers with maxpool
    conv1 = create_conv_layer(inputs, n_base_filters, kernel_size, stride, activation, padding)
    conv2 = create_conv_layer(conv1, n_base_filters * 2, kernel_size, stride, activation, padding)
    maxpool1 = MaxPooling3D(pool_size=2,)(conv2)

    conv3 = create_conv_layer(maxpool1, n_base_filters * 2, kernel_size, stride, activation, padding)
    conv4 = create_conv_layer(conv3, n_base_filters * 4, kernel_size, stride, activation, padding)
    maxpool2 = MaxPooling3D(pool_size=2)(conv4)

    conv5 = create_conv_layer(maxpool2, n_base_filters * 4, kernel_size, stride, activation, padding)
    conv6 = create_conv_layer(conv5, n_base_filters * 8, kernel_size, stride, activation, padding)
    maxpool3 = MaxPooling3D(pool_size=2)(conv6)

    #Layers with upsampling
    # You can either use deconvolution or upsampling.  It seems that the code
    # from github uses deconvolution.
    conv7 = create_conv_layer(maxpool3, n_base_filters * 8, kernel_size, stride, activation, padding)
    conv8 = create_conv_layer(conv7, n_base_filters * 16, kernel_size, stride, activation, padding)
    upsampling1 = get_upconvolution(n_base_filters * 16, use_upsampling)(conv8)
    concat1 = keras.layers.concatenate([upsampling1 ,conv6])

    conv9 = create_conv_layer(concat1, n_base_filters * 8, kernel_size, stride, activation, padding)
    conv10 = create_conv_layer(conv9, n_base_filters * 8, kernel_size, stride, activation, padding)
    upsampling2 = get_upconvolution(n_base_filters * 8, use_upsampling)(conv10)
    concat2 = keras.layers.concatenate([upsampling2 ,conv4])

    conv11 = create_conv_layer(concat2, n_base_filters * 4, kernel_size, stride, activation, padding)
    conv12 = create_conv_layer(conv11, n_base_filters * 4, kernel_size, stride, activation, padding)
    upsampling3 = get_upconvolution(n_base_filters * 4, use_upsampling)(conv12)
    concat3 = keras.layers.concatenate([upsampling3 ,conv2])

    # Final layers
    conv13 = create_conv_layer(concat3, n_base_filters * 2, kernel_size, stride, activation, padding)
    conv14 = create_conv_layer(conv13, n_base_filters * 2, kernel_size, stride, activation, padding)

    # TODO: is kernel size 1 here?
    conv15 = Conv3D(filters = 2, kernel_size = 1, strides = stride)(conv14)
    act = Activation('softmax')(conv15)
    model = Model(inputs = inputs, outputs = act)
    
    # TODO: Remember to use different loss function.
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss = 'kld', metrics = ['accuracy'])
    print(initial_learning_rate)
    print(model.summary())
    return model

def create_conv_layer(input_layer, n_filters, kernel_size, stride, activation, padding, uses_batch_normalization=False):
    conv_layer = Conv3D(filters=n_filters, kernel_size=kernel_size, strides=stride, activation=activation, padding=padding)(input_layer)
    if(uses_batch_normalization):
        conv_layer = BatchNormalization(axis=1)(conv_layer)
    
    return conv_layer

def get_upconvolution(nfilters, use_upsampling, kernel_size=2):
    if(use_upsampling):
        return UpSampling3D(size=kernel_size)
    else:
        return Conv3DTranspose(filters=nfilters, kernel_size=kernel_size, strides=2)

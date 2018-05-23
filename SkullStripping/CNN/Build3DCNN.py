import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.layers import Activation
from keras.engine import Input, Model

from keras.optimizers import Adam
from keras.layers.convolutional import Conv3D, MaxPooling3D
from extra import dice_coefficient_loss

def build_3DCNN(input_shape, gpus, loss_function, pool_size=(2, 2, 2),
                  initial_learning_rate=0.00001, deconvolution=False, stride=1):
    inputs = Input(input_shape)
    activation = 'relu'

    conv1 = Conv3D(filters=16, kernel_size=(4, 4, 4), strides=stride, padding='valid')(inputs)
    act1 = Activation(activation=activation)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(act1)
    conv2 = Conv3D(filters=24, kernel_size=(5, 5, 5), strides=stride, padding='valid')(pool1)
    act2 = Activation(activation=activation)(conv2)
    conv3 = Conv3D(filters=28, kernel_size=(5, 5, 5), strides=stride, padding='valid')(act2)
    act3 = Activation(activation=activation)(conv3)
    conv4 = Conv3D(filters=34, kernel_size=(5, 5, 5), strides=stride, activation=activation, padding='valid')(act3)
    conv5 = Conv3D(filters=42, kernel_size=(5, 5, 5), strides=stride, activation=activation, padding='valid')(conv4)
    conv6 = Conv3D(filters=50, kernel_size=(5, 5, 5), strides=stride, activation=activation, padding='valid')(conv5)
    conv7 = Conv3D(filters=50, kernel_size=(5, 5, 5), strides=stride, activation=activation, padding='valid')(conv6)

    conv8 = Conv3D(filters=2, kernel_size=(1, 1, 1))(conv7)
    act = Activation('softmax')(conv8)

    parallel_model = None
    if(loss_function is None):
        print("Using kld function")
        loss_function = 'kld'
    elif(loss_function == 'dcl'):
        print("Using dice loss function")
        loss_function = dice_coefficient_loss

    if(gpus > 1):
        with tf.device('/cpu:0'):
            model = Model(inputs = inputs, outputs = act)
            
        parallel_model = multi_gpu_model(model, gpus)
        parallel_model.compile(optimizer=Adam(lr=initial_learning_rate), loss=loss_function, metrics=['accuracy'])
    else:
        model = Model(inputs = inputs, outputs = act)
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=loss_function, metrics=['accuracy'])
    
    print(model.summary())
    return model, parallel_model
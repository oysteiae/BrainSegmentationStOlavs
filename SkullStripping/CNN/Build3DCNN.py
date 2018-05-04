import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.layers import Activation
from keras.engine import Input, Model

from keras.optimizers import Adam
from keras.layers.convolutional import Conv3D, MaxPooling3D
from extra import dice_coefficient_loss

def build_3DCNN(input_shape, gpus, pool_size=(2, 2, 2),
                  initial_learning_rate=0.00001, deconvolution=False, stride=1, using_sparse_categorical_crossentropy=False):
    inputs = Input(input_shape)
    activation = 'relu'

    conv1 = Conv3D(filters=16, kernel_size=(4, 4, 4), strides=stride, activation=activation, padding='valid')(inputs)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(filters=24, kernel_size=(5, 5, 5), strides=stride, activation=activation, padding='valid')(pool1)
    conv3 = Conv3D(filters=28, kernel_size=(5, 5, 5), strides=stride, activation=activation, padding='valid')(conv2)
    conv4 = Conv3D(filters=34, kernel_size=(5, 5, 5), strides=stride, activation=activation, padding='valid')(conv3)
    conv5 = Conv3D(filters=42, kernel_size=(5, 5, 5), strides=stride, activation=activation, padding='valid')(conv4)
    conv6 = Conv3D(filters=50, kernel_size=(5, 5, 5), strides=stride, activation=activation, padding='valid')(conv5)
    conv7 = Conv3D(filters=50, kernel_size=(5, 5, 5), strides=stride, activation=activation, padding='valid')(conv6)

    conv8 = Conv3D(filters=2, kernel_size=(1, 1, 1))(conv7)
    act = Activation('softmax')(conv8)

    parallel_model = None

    if using_sparse_categorical_crossentropy:
        print("Using sparse categorical crossentropy as loss function")
        #categorical_labels = to_categorical(int_labels, num_classes=None)
        if(gpus > 1):
            with tf.device('/cpu:0'):
                model = Model(inputs = inputs, outputs = act)
            
            parallel_model = multi_gpu_model(model, gpus)
            parallel_model.compile(optimizer=Adam(lr=initial_learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:
            model = Model(inputs = inputs, outputs = act)
            model.compile(optimizer=Adam(lr=initial_learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        print("Using Kullback-Leibler divergence as loss function")
        if(gpus > 1):
            with tf.device('/cpu:0'):
                model = Model(inputs = inputs, outputs = act)
            
            parallel_model = multi_gpu_model(model, gpus)
            parallel_model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=['accuracy'])
        else:
            model = Model(inputs = inputs, outputs = act)
            model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=['accuracy'])
    
    print(model.summary())
    return model, parallel_model
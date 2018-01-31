import numpy as np
import keras
from keras.engine import Input, Model
from keras.layers import Activation, BatchNormalization
from keras.layers.convolutional import Conv3D, MaxPooling3D, Deconv3D, UpSampling3D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from extra import dice_coefficient_loss

class Trainer3DUnet:
    """description of class"""
    def __init__(self):
        model = self.build_3dUnet()

    # Need printing to get what's happening
    def build_3dUnet(self, input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                     depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False,
                     batch_normalization=False, activation_name="sigmoid"):
        inputs = Input(input_shape)
        current_layer = inputs
        levels = list()

        for layer_depth in range(depth):
            # Creates two layers
            layer1 = self.create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                              batch_normalization=batch_normalization)
            layer2 = self.create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                              batch_normalization=batch_normalization)
            
            # Why current_layer?
            # Every layer except the last one has max pooling 
            if layer_depth < depth - 1:
                current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
                levels.append([layer1, layer2, current_layer])
            else:
                current_layer = layer2
                levels.append([layer1, layer2])

        # add levels with up-convolution or up-sampling
        for layer_depth in range(depth-2, -1, -1):
            # Is this possible because of the functinoal API?
            up_convolution = self.get_up_convolution(pool_size=pool_size, 
                                                     deconvolution=deconvolution,
                                                     n_filters=current_layer._keras_shape[1])(current_layer)
            concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
                
            current_layer = self.create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                          input_layer=concat,
                                                          batch_normalization=batch_normalization)
            current_layer = self.create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                          input_layer=current_layer,
                                                          batch_normalization=batch_normalization)

        final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
        act = Activation('softmax')(final_convolution) # I think softmax is in the last layer.
        model = Model(inputs=inputs, outputs=act)

        # You have to make the dice coefficient loss function
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
            
        print(model.summary())
        return model

    def create_convolution_block(self, 
                                    input_layer, 
                                    n_filters, 
                                    kernel=(3, 3, 3), 
                                    padding='same', 
                                    strides=(1, 1, 1)):
        # Can add batch normilization or instance normalization later. Maybe not since training time should not be an issue.
        return Conv3D(n_filters, kernel, padding=padding, strides=strides, Activation='relu')(input_layer) # Maybe sigmoid here instead

    # What exactly does this do?
    def compute_level_output_shape(self, 
                                    n_filters, 
                                    depth, 
                                    pool_size, 
                                    image_shape):
        """
        Each level has a particular output shape based on the number of filters used in that level and the depth or number 
        of max pooling operations that have been done on the data at that point.
        :param image_shape: shape of the 3d image.
        :param pool_size: the pool_size parameter used in the max pooling operation.
        :param n_filters: Number of filters used by the last node in a given level.
        :param depth: The number of levels down in the U-shaped model a given node is.
        :return: 5D vector of the shape of the output node 
        """
        output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
        return tuple([None, n_filters] + output_image_shape)


    def get_up_convolution(self, 
                            n_filters, 
                            pool_size, 
                            kernel_size=(2, 2, 2), 
                            strides=(2, 2, 2),
                            deconvolution=False):
        if deconvolution:
            return Deconv3D(filters=n_filters, 
                            kernel_size=kernel_size,
                            strides=strides)
        else:
            return UpSampling3D(size=pool_size) # Same as unpooling. So this is the reverse of maxpooling.
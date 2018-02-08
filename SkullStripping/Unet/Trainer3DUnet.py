from Unet.Build3DUnet import build_3DUnet
import Trainer
import helper
import numpy as np

class Trainer3DUnet:
    """Class for training a 3D Unet"""
    def __init__(self, input_shape, using_sparse_categorical_crossentropy=False):
        self.input_shape = input_shape
        self.using_sparse_categorical_crossentropy = using_sparse_categorical_crossentropy
    
    def build_model(self, using_sparse_categorical_crossentropy=False):
        return build_3DUnet(self.input_shape)

    def train(self, data_file_location, label_file_location, n_epochs, save_name, batch_size=4, use_cross_validation=False, validation_label_location="", validation_data_location=""):
        # Loads the files
        d = helper.load_files(data_file_location)
        l = helper.load_files(label_file_location)
        training_data, training_labels = helper.patchCreator(d, l)

        if(use_cross_validation):
            Trainer.train_crossvalidation(self, training_data, training_labels, n_epochs, save_name, batch_size, using_sparse_categorical_crossentropy=self.using_sparse_categorical_crossentropy)
        else:
            if(validation_data_location != ""):
                validation_d = helper.load_files([validation_data_location])
                validation_l = helper.load_files([validation_labels_location])
                validation_data, validation_labels = helper.patchCreator(validation_d, validation_l)
                Trainer.train_without_crossvalidation(self, training_data, training_labels, n_epochs, save_name, batch_size, self.using_sparse_categorical_crossentropy, validation_data, validation_labels)
            else:
                Trainer.train_without_crossvalidation(self, training_data, training_labels, n_epochs, save_name, batch_size, self.using_sparse_categorical_crossentropy)

    # Don't know if I need get_cubes or if I should just return the full image.
    def get_generator(self, data, labels, mini_batch_size=4, using_sparse_categorical_crossentropy=False):
        while True:
            x_list = np.zeros((mini_batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2], 1))
            y_list = np.zeros((mini_batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2], 1))
        
            for i in range(mini_batch_size):
                dat, lab = self.get_cubes(data, labels, 0, len(data))
                x_list[i] = dat
                y_list[i] = lab
             
            yield (x_list, y_list)

    def get_cubes(self, data, labels, i_min, i_max):
        i = np.random.randint(i_min, i_max) # Used for selecting a random example
        # Something was wrong with the shape here.
        return data[i], np.expand_dims(labels[i], axis=3)
from Unet.Build3DUnet import build_3DUnet
import Trainer
import helper
import numpy as np
from Callbacks.Logger import LossHistory
from Callbacks.MonitorStopping import MonitorStopping
from keras.callbacks import ModelCheckpoint
import h5py
from os import mkdir

class Trainer3DUnet:
    """Class for training a 3D Unet"""
    def __init__(self, input_shape, gpus, training_with_slurm=False, loss_function='kld'):
        self.input_shape = input_shape
        self.gpus = gpus
        self.loss_function = loss_function
        # Have to have this since saving the multi gpu model is not loadable on single gpu instances
        # The weights are shared so it should work
        self.model_for_saving_weights = None
        self.training_with_slurm=training_with_slurm
    
    def build_model(self):
        if(self.gpus == 1):
            model, parallel_model =  build_3DUnet(self.input_shape, self.gpus, self.loss_function)
            self.model_for_saving_weights = model
            return model
        else:
            model, parallel_model =  build_3DUnet(self.input_shape, self.gpus, self.loss_function)
            self.model_for_saving_weights = model
            return parallel_model

    def train(self, data_file_location, label_file_location, n_epochs, save_name, batch_size=4, use_cross_validation=False, use_validation=False, training_with_slurm=False, validation_data=None, validation_labels=None, location_previous_training_and_validation_indices=None):
        # Loads the files
        d = np.asarray(helper.load_files(data_file_location))
        l = np.asarray(helper.load_files(label_file_location))

        if(use_cross_validation):
            training_data, training_labels = helper.patchCreator(d, l, normalize=True, save_name=save_name)
            Trainer.train_crossvalidation(self, training_data, training_labels, n_epochs, save_name, batch_size)
        else:
            Trainer.train_without_crossvalidation(self, d, l, n_epochs, save_name, use_validation=use_validation, training_with_slurm=training_with_slurm, validation_data_location=validation_data, validation_labels_location=validation_labels, location_previous_training_and_validation_indices=location_previous_training_and_validation_indices)

    def get_callbacks(self, model_save_name, model, save_name):
        # Callback methods
        #checkpoint = ModelCheckpoint(model_save_name, monitor='loss', verbose=1, save_best_only=False, mode='min', period=100)
        logger = LossHistory()
        decrease_learning_rate_callback = MonitorStopping(model)
        if(self.gpus == 1):
            if(self.training_with_slurm == False):
                parentDirectory = helper.get_parent_directory()
                experiment_directory = parentDirectory + "/Experiments/" + save_name + "/"
                try:    
                    mkdir(experiment_directory)
                except FileExistsError:
                    print("Folder exists, do nothing")
                
                checkpoint = ModelCheckpoint(experiment_directory +  model_save_name, monitor='loss', verbose=1, save_best_only=False, mode='min', period=100)
                return [logger, checkpoint, decrease_learning_rate_callback]
            else:
                try:
                    mkdir("/home/oysteiae/Experiments/" + save_name + "/")        
                except FileExistsError:
                    print("Folder exists, do nothing")
                
                checkpoint = ModelCheckpoint("/home/oysteiae/Experiments/" + save_name + "/" + model_save_name, monitor='loss', verbose=1, save_best_only=False, mode='min', period=100)
                return [logger, checkpoint, decrease_learning_rate_callback]
        else:
            return [logger, decrease_learning_rate_callback]

    # Don't know if I need get_cubes or if I should just return the full image.
    def get_generator(self, data, labels, mini_batch_size=4):
        while True:
            x_list = np.zeros((mini_batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2], 1))
            y_list = np.zeros((mini_batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2], 2))
            for i in range(mini_batch_size):
                dat, lab = self.get_cubes(data, labels, 0, len(data))
                dat = self.data_augmentation_greyvalue(dat)
                
                x_list[i] = dat
                y_list[i] = lab
            yield (x_list, y_list)

    def get_cubes(self, data, labels, i_min, i_max):
        i = np.random.randint(i_min, i_max) # Used for selecting a random example
        dat = np.zeros((1, self.input_shape[0], self.input_shape[1], self.input_shape[2], 1), dtype="float32")
        lab = np.zeros((1, self.input_shape[0], self.input_shape[1], self.input_shape[2], 2), dtype="int16")
        data_shape = data[i].shape #shape = (176, 208, 176, 1)

        off = [np.random.randint(0, data_shape[x] - self.input_shape[x]) for x in range(0, 3)]
        dat[0,...] = data[i][off[0] : off[0] + self.input_shape[0], off[1] : off[1] + self.input_shape[1], off[2] : off[2] + self.input_shape[2], :]
        lab[0, :, :, :, 0] = labels[i][off[0] : off[0] + self.input_shape[0], off[1] : off[1] + self.input_shape[1], off[2] : off[2] + self.input_shape[2]] 
        lab[0, :, :, :, 1] = (labels[i][off[0] : off[0] + self.input_shape[0], off[1] : off[1] + self.input_shape[1], off[2] : off[2] + self.input_shape[2]] < 1).astype('int8')

        return dat, lab

    # Taken from: https://github.com/GUR9000/Deep_MRI_brain_extraction
    def data_augmentation_greyvalue(self, dat, max_shift=0.05, max_scale=1.3, min_scale=0.85):
        sh = (0.5 - np.random.random()) * max_shift * 2.
        scale = (max_scale - min_scale) * np.random.random() + min_scale
        return (sh + dat * scale).astype("float32")
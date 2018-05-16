from numpy.random import seed
from Callbacks.Logger import LossHistory
from Callbacks.MonitorStopping import MonitorStopping
from keras.callbacks import EarlyStopping, ModelCheckpoint
import helper
import numpy as np
from CNN.Build3DCNN import build_3DCNN
import Trainer
from CNN.helper_methods_CNN import compute_label_offset
from os import mkdir

class Trainer3DCNN:
    'Class used for training a 3D CNN for predicting MRI images'
    def __init__(self, gpus, cnn_input_size=(59, 59, 59, 1), using_sparse_categorical_crossentropy=False, training_with_slurm=False, loss_function='kld'):
        # TODO: determine input shape based on what you're training on.
        self.cnn_input_size = cnn_input_size
        self.gpus = gpus
        self.model_for_saving_weights = None
        self.training_with_slurm=training_with_slurm
        self.loss_function = loss_function
        self.labels_offset = compute_label_offset()
    
    # Output size of the model should really not be computed here. It shuold be computed in the constructor, but because of the way it's implemented right now I'll leave it
    def build_model(self,using_sparse_categorical_crossentropy=False):
        if(self.gpus == 1):
            model, parallel_model = build_3DCNN(self.cnn_input_size, self.gpus, self.loss_function)
            self.model_for_saving_weights = model
            self.output_size = self.model_for_saving_weights.layers[-1].output_shape
            return model
        else:
            model, parallel_model = build_3DCNN(self.cnn_input_size, self.gpus, self.loss_function)
            self.model_for_saving_weights = model
            self.output_size = self.model_for_saving_weights.layers[-1].output_shape
            return parallel_model

    def train(self, data_file_location, label_file_location, n_epochs, save_name, batch_size=4, use_cross_validation=False, use_validation=False, training_with_slurm=False, validation_data=None, validation_labels=None):
        # Loads the files
        d = np.asarray(helper.load_files(data_file_location))
        l = np.asarray(helper.load_files(label_file_location))
        
        if(use_cross_validation):
            training_data, training_labels = helper.patchCreator(d, l, normalize=True, save_name=save_name)
            Trainer.train_crossvalidation(self, training_data, training_labels, n_epochs, save_name, batch_size)
        else:
            Trainer.train_without_crossvalidation(self, d, l, n_epochs, save_name, use_validation=use_validation, training_with_slurm=training_with_slurm, validation_data_location=validation_data, validation_labels_location=validation_labels)

    def get_callbacks(self, model_save_name, model, save_name):
        # Callback methods
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
                checkpoint = ModelCheckpoint(model_save_name, monitor='loss', verbose=1, save_best_only=False, mode='min', period=100)
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

    def get_generator(self, data, labels, mini_batch_size=4, using_sparse_categorical_crossentropy=False):
        while True:
            # Find a way to use input_size and output_size here.
            x_list = np.zeros((mini_batch_size, self.cnn_input_size[0], self.cnn_input_size[1], self.cnn_input_size[2], 1))
            y_list = np.zeros((mini_batch_size, self.output_size[1], self.output_size[2], self.output_size[3], self.output_size[4]))
        
            for i in range(mini_batch_size):
                dat, lab = self.get_cubes(data, labels, 0, len(data), self.cnn_input_size[0])
                dat = self.data_augmentation_greyvalue(dat)
                x_list[i] = dat
                y_list[i] = lab
             
            yield (x_list, y_list)
    
    def get_cubes(self, data, labels, i_min, i_max, input_size, stride=2):
        i = np.random.randint(i_min, i_max) # Used for selecting a random example
        dat = np.zeros((1, input_size, input_size, input_size, 1), dtype="float32")
        labshape = ((1,) + (self.output_size[1],) * 3 + (2,)) #ndim
        lab = np.zeros(labshape, dtype="int16")
        data_shape = data[i].shape

        off = [np.random.randint(0, data_shape[x] - input_size) for x in range(0, 3)]
        loff = tuple(off) + self.labels_offset
    
        dat[0,...] = data[i][off[0] : off[0] + input_size, off[1] : off[1] + input_size, off[2] : off[2] + input_size, :]
    
        lab[0, :, :, :, 0] = labels[i][loff[0] : loff[0] + self.output_size[1] * stride : stride, loff[1] : loff[1] + self.output_size[2] * stride : stride, loff[2]:loff[2] + self.output_size[3] * stride : stride]
        lab[0, :, :, :, 1] = (labels[i][loff[0] : loff[0] + self.output_size[1] * stride : stride, loff[1] : loff[1] + self.output_size[2] * stride : stride, loff[2]:loff[2] + self.output_size[3] * stride : stride] < 1).astype('int8')

        # Returns cubes of the training data
        return dat, lab

    # Taken from: https://github.com/GUR9000/Deep_MRI_brain_extraction
    def data_augmentation_greyvalue(self, dat, max_shift=0.05, max_scale=1.3, min_scale=0.85):
        sh = (0.5 - np.random.random()) * max_shift * 2.
        scale = (max_scale - min_scale) * np.random.random() + min_scale
        return (sh + dat * scale).astype("float32")

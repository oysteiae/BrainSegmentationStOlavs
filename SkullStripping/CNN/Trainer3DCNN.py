from numpy.random import seed
from Callbacks import Logger
from Callbacks import MonitorStopping
from sklearn.cross_validation import KFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
import helper
import numpy as np
from CNN.Build3DCNN import build_3DCNN
import Trainer

class Trainer3DCNN:
    'Class used for training a 3D CNN for predicting MRI images'
    def __init__(self, cnn_input_size=(59, 59, 59, 1), using_sparse_categorical_crossentropy=False):
        # TODO: determine input shape based on what you're training on.
        self.cnn_input_size = cnn_input_size
        self.using_sparse_categorical_crossentropy = using_sparse_categorical_crossentropy
    
    def build_model(self):
        return build_3DCNN(self.cnn_input_size, using_sparse_categorical_crossentropy=self.using_sparse_categorical_crossentropy)

    def train(data_file_location, label_file_location, n_epochs, save_name, batch_size=4, use_cross_validation=False, validation_label_location="", validation_data_location=""):
        # Loads the files
        d = helper.load_files(data_file_location)
        l = helper.load_files(label_file_location)
        training_data, training_labels = helper.patchCreator(d, l)

        if(use_cross_validation):
            Trainer.train_crossvalidation(self, training_data, training_labels, n_epochs, save_name, batch_size, self.using_sparse_categorical_crossentropy)
        else:
            if(validation_data_location != ""):
                validation_d = helper.load_files([validation_data_location])
                validation_l = helper.load_files([validation_labels_location])
                validation_data, validation_labels = helper.patchCreator(validation_d, validation_l)
                Trainer.train_without_crossvalidation(self, training_data, training_labels, n_epochs, save_name, batch_size, self.using_sparse_categorical_crossentropy, validation_data, validation_labels)
            else:
                Trainer.train_without_crossvalidation(self, training_data, training_labels, n_epochs, save_name, batch_size, self.using_sparse_categorical_crossentropy)

    # def get_generator(data, labels, mini_batch_size=4):
    # TODO: maybe add augmentation in the long run
    # What does even mini_batch_size do here if you set it in the model.fit()?
    def get_generator(self, data, labels, input_size, output_size, mini_batch_size=4):
        while True:
            # Find a way to use input_size and output_size here.
            x_list = np.zeros((mini_batch_size, 59, 59, 59, 1))
            if(self.using_sparse_categorical_crossentropy):
                y_list = np.zeros((mini_batch_size, 4, 4, 4, 1))
            else:
                y_list = np.zeros((mini_batch_size, 4, 4, 4, 2))
        
            for i in range(mini_batch_size):
                #(data, labels, i_min, i_max, input_size,
                #number_of_labeled_points_per_dim=4, stride=2,
                #labels_offset=[26, 26, 26]
                dat, lab = get_cubes(data, labels, 0, len(data), 59)
                dat = data_augmentation_greyvalue(dat)
                x_list[i] = dat
                y_list[i] = lab
             
            yield (x_list, y_list)

    # Need to rethink now that unet can also use this.
    # TODO: should compute number_of_labeled_points_per_dim myself
    # It's computed: ((Input_voxels_shape - 53)/2) + 1
    # The problems with the shapes:
    # https://github.com/fchollet/keras/issues/4781
    # Taken from: https://github.com/GUR9000/Deep_MRI_brain_extraction
    def get_cubes(self, data, labels, i_min, i_max, input_size, number_of_labeled_points_per_dim=4, stride=2):
        labels_offset = np.array((26, 26, 26))

        i = np.random.randint(i_min, i_max) # Used for selecting a random example
        dat = np.zeros((1, input_size, input_size, input_size, 1), dtype="float32")
        labshape = ((1,) + (number_of_labeled_points_per_dim,) * 3 + (2,)) #ndim
        lab = np.zeros(labshape, dtype="int16")
        data_shape = data[i].shape #shape = (176, 208, 176, 1)

        off = [np.random.randint(0, data_shape[x] - input_size) for x in range(0, 3)]
        loff = tuple(off) + labels_offset #shape = (88, 146, 67)
    
        dat[0,...] = data[i][off[0] : off[0] + input_size, off[1] : off[1] + input_size, off[2] : off[2] + input_size, :] #shape = (59, 59, 59, 1)
    
        if self.using_sparse_categorical_crossentropy:
            lab = labels[i][loff[0] : loff[0] + number_of_labeled_points_per_dim * stride : stride, loff[1] : loff[1] + number_of_labeled_points_per_dim * stride : stride, loff[2]:loff[2] + number_of_labeled_points_per_dim * stride : stride] #shape = (4, 4, 4)
            lab = np.expand_dims(lab, axis=0)
            lab = np.expand_dims(lab, -1)
        else:
            lab[0, :, :, :, 0] = labels[i][loff[0] : loff[0] + number_of_labeled_points_per_dim * stride : stride, loff[1] : loff[1] + number_of_labeled_points_per_dim * stride : stride, loff[2]:loff[2] + number_of_labeled_points_per_dim * stride : stride] #shape = (4, 4, 4)
            lab[0, :, :, :, 1] = (labels[i][loff[0] : loff[0] + number_of_labeled_points_per_dim * stride : stride, loff[1] : loff[1] + number_of_labeled_points_per_dim * stride : stride, loff[2]:loff[2] + number_of_labeled_points_per_dim * stride : stride] < 1).astype('int8') #shape = (4, 4, 4)

        # TODO: do you need these extra dims?
        #dat = np.expand_dims(dat, axis=0) #shape = (1, 59, 59, 59, 1)

        # Returns cubes of the training data
        return dat, lab

    # Taken from: https://github.com/GUR9000/Deep_MRI_brain_extraction
    def data_augmentation_greyvalue(self, dat, max_shift=0.05, max_scale=1.3, min_scale=0.85):
        sh = (0.5 - np.random.random()) * max_shift * 2.
        scale = (max_scale - min_scale) * np.random.random() + min_scale
        return (sh + dat * scale).astype("float32")
from Build3DUnet import build_3DUnet
import Trainer
import helper

class Trainer3DUnet:
    """description of class"""
    def __init__(self, input_shape, using_sparse_categorical_crossentropy=False):
        self.test = ""
        self.input_shape = input_shape
        self.using_sparse_categorical_crossentropy = using_sparse_categorical_crossentropy

    def build_model(self):
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
                Trainer.train_without_crossvalidation(self, training_data, training_labels, n_epochs, save_name, batch_size, using_sparse_categorical_crossentropy, validation_data, validation_labels)
            else:
                Trainer.train_without_crossvalidation(self, training_data, training_labels, n_epochs, save_name, batch_size, using_sparse_categorical_crossentropy)

    # Don't know if I need get_cubes or if I should just return the full image.
    def get_generator(self, data, labels, input_size, output_size, mini_batch_size=4):
        while True:
            # Find a way to use input_size and output_size here.
            x_list = np.zeros((mini_batch_size, 59, 59, 59, 1))
            if(self.using_sparse_categorical_crossentropy):
                y_list = np.zeros((mini_batch_size, 4, 4, 4, 1))
            else:
                y_list = np.zeros((mini_batch_size, 4, 4, 4, 2))
        
            for i in range(mini_batch_size):
                dat, lab = get_cubes(data, labels, 0, len(data), 59)
                dat = data_augmentation_greyvalue(dat)
                x_list[i] = dat
                y_list[i] = lab
             
            yield (x_list, y_list)

    def get_cubes(self, data, labels, i_min, i_max, input_size, number_of_labeled_points_per_dim=4, stride=2):
        dat = np.zeros((1, input_size, input_size, input_size, 1), dtype="float32")
        labshape = ((1,) + (number_of_labeled_points_per_dim,) * 3 + (2,)) #ndim
        lab = np.zeros(labshape, dtype="int16")

        return dat, lab
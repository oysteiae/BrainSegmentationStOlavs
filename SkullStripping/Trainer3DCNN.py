from Logger import LossHistory
from numpy.random import seed
from MonitorStopping import MonitorStopping
from sklearn.cross_validation import KFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
import helper
import numpy as np

class Trainer3DCNN:
    'Class used for training a 3D CNN for predicting MRI images'
    def __init__(self, data_file_location, label_file_location, save_name, using_sparse_categorical_crossentropy=False, use_cross_validation=False):
        #Parameters
        self.initial_learning_rate = 0.00001
        self.n_epochs = 50000000000000000
        #n_epochs = 500
        self.batch_size = 4
        self.save_name = save_name
        self.using_sparse_categorical_crossentropy = using_sparse_categorical_crossentropy
        self.use_cross_validation = use_cross_validation

        # TODO: determine input shape based on what you're training on.
        self.cnn_input_size = (59, 59, 59, 1)

        # Loads the files
        d = helper.load_files(data_file_location)
        l = helper.load_files(label_file_location)
        self.training_data, self.training_data_labels = helper.patchCreator(d, l)
    
    def train_crossvalidation(build_CNN):
        j = 1
        seed = 7
        kfold = KFold(n=len(training_data),n_folds=2, random_state=seed, shuffle=True)
        for train, test in kfold:
            print("Testing on", helper.list_to_string(train))    
            print("Testing on", helper.list_to_string(test))
    
            model = None
            model = build_CNN(input_shape=selfcnn_input_size, using_sparse_categorical_crossentropy=self.using_sparse_categorical_crossentropy)
        
            training_generator = self.get_generator(self.training_data[train], self.training_data_labels[train], mini_batch_size=self.batch_size, using_sparse_categorical_crossentropy=self.using_sparse_categorical_crossentropy)
            validation_generator = self.get_generator(self.training_data[test], self.training_data_labels[test], mini_batch_size=self.batch_size, using_sparse_categorical_crossentropy=self.using_sparse_categorical_crossentropy)
            
            model_save_name = self.save_name + str(j) + ".h5"
    
            # Callback methods
            checkpoint = ModelCheckpoint(model_save_name, monitor='loss', verbose=1, save_best_only=False, mode='min', period=100)
            logger = LossHistory()
            decrease_learning_rate_callback = MonitorStopping(model)

            callbacks = [checkpoint, logger, decrease_learning_rate_callback]
            self.train_net(model, training_generator, validation_generator, self.n_epochs, callbacks)
        
            logs_save_name = self.save_name + "_logs" + str(j)
            helper.save(model_save_name, logs_save_name, logger, model)
            j += 1

    def train_without_crossvalidation(build_CNN, validation_data_location="", validation_labels_location=""):
        model = build_CNN(input_shape=self.cnn_input_size, using_sparse_categorical_crossentropy=self.using_sparse_categorical_crossentropy)
        training_generator = self.get_generator(self.training_data, self.training_data_labels, mini_batch_size=self.batch_size, using_sparse_categorical_crossentropy=self.using_sparse_categorical_crossentropy)
        
        if(validation_data_location != ""):
            validation_d = helper.load_files([validation_data_location])
            validation_l = helper.load_files([validation_labels_location])
            validation_data, validation_data_labels = helper.patchCreator(validation_d, validation_l)
            validation_generator = self.get_generator(validation_data, validation_data_labels, mini_batch_size=batch_size, using_sparse_categorical_crossentropy=using_sparse_categorical_crossentropy)
        
        model_save_name = save_name + ".h5"
    
        # Callback methods
        checkpoint = ModelCheckpoint(model_save_name, monitor='loss', verbose=1, save_best_only=False, mode='min', period=100)
        logger = LossHistory()
        decrease_learning_rate_callback = MonitorStopping(model)

        callbacks = [checkpoint, logger, decrease_learning_rate_callback]
        
        #def train(model, training_generator, n_epochs, callbacks, using_sparse_categorical_crossentropy=False):
        if(validation_data_location != ""):
            self.train_net(model, training_generator, validation_generator, self.n_epochs, callbacks, self.using_sparse_categorical_crossentropy, uses_validation_data=(validation_data_location != ""))
        else:
            self.train_net(model, training_generator, None, self.n_epochs, callbacks, self.using_sparse_categorical_crossentropy, uses_validation_data=(validation_data_location != ""))
        
        logs_save_name = save_name + "_logs"
        save(model_save_name, logs_save_name, logger, model)

    def train_net(self, model, training_generator, validation_generator, n_epochs, callbacks, using_sparse_categorical_crossentropy=False, uses_validation_data=True):
        #Should perhaps set steps_per_epoch to 1
        if(using_sparse_categorical_crossentropy):
            model.fit_generator(
                generator=training_generator,
                validation_data = validation_generator,
                validation_steps = 1,
                steps_per_epoch= 1,#len(training_data)/batch_size,
                epochs=n_epochs,
                pickle_safe=False,
                verbose=2,
                callbacks=callbacks)
        if(uses_validation_data):
            model.fit_generator(
                generator=training_generator,
                validation_data = validation_generator,
                validation_steps = 1,
                steps_per_epoch=1,
                epochs=n_epochs,
                pickle_safe=False,
                verbose=0,
                callbacks=callbacks)
        else:
            model.fit_generator(
                generator=training_generator,
                steps_per_epoch=1,
                epochs=n_epochs,
                pickle_safe=False,
                verbose=0,
                callbacks=callbacks)

    # def get_generator(data, labels, mini_batch_size=4):
    # TODO: maybe add augmentation in the long run
    # What does even mini_batch_size do here if you set it in the model.fit()?
    def get_generator(self, data, labels, mini_batch_size=4, using_sparse_categorical_crossentropy=False):
        while True:
            x_list = np.zeros((mini_batch_size, 59, 59, 59, 1))
            if(using_sparse_categorical_crossentropy):
                y_list = np.zeros((mini_batch_size, 4, 4, 4, 1))
            else:
                y_list = np.zeros((mini_batch_size, 4, 4, 4, 2))
        
            for i in range(mini_batch_size):
                #(data, labels, i_min, i_max, input_size,
                #number_of_labeled_points_per_dim=4, stride=2, labels_offset=[26, 26, 26]
                dat, lab = self.get_cubes(data, labels, 0, len(data), 59, using_sparse_categorical_crossentropy=using_sparse_categorical_crossentropy)
                dat = self.data_augmentation_greyvalue(dat)
                x_list[i] = dat
                y_list[i] = lab
             
            yield (x_list, y_list)

    # TODO: should compute number_of_labeled_points_per_dim myself
    # It's computed: ((Input_voxels_shape - 53)/2) + 1
    # The problems with the shapes: https://github.com/fchollet/keras/issues/4781
    # Taken from: https://github.com/GUR9000/Deep_MRI_brain_extraction
    def get_cubes(self, data, labels, i_min, i_max, input_size, number_of_labeled_points_per_dim=4, stride=2, using_sparse_categorical_crossentropy=False):
        labels_offset = np.array((26, 26, 26))

        i = np.random.randint(i_min, i_max) # Used for selecting a random example
        dat = np.zeros((1, input_size, input_size, input_size, 1), dtype="float32")
        labshape = ((1,) + (number_of_labeled_points_per_dim,) * 3 + (2,)) #ndim
        lab = np.zeros(labshape, dtype="int16")
        data_shape = data[i].shape #shape = (176, 208, 176, 1)

        off = [np.random.randint(0, data_shape[x] - input_size) for x in range(0, 3)]
        loff = tuple(off) + labels_offset #shape = (88, 146, 67)
    
        dat[0,...] = data[i][off[0] : off[0] + input_size, off[1] : off[1] + input_size, off[2] : off[2] + input_size, :] #shape = (59, 59, 59, 1)
    
        if using_sparse_categorical_crossentropy:
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
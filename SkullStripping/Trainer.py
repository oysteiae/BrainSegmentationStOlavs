from Logger import LossHistory
from numpy.random import seed
from MonitorStopping import MonitorStopping
from sklearn.cross_validation import KFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
import helper
import numpy as np

def train_crossvalidation(neural_net, training_data, training_data_labels, n_epochs, cnn_input_size):
    j = 1
    seed = 7
    kfold = KFold(n=len(training_data),n_folds=2, random_state=seed, shuffle=True)
    for train, test in kfold:
        print("Testing on", helper.list_to_string(train))    
        print("Testing on", helper.list_to_string(test))
    
        model = None
        model = neural_net.build_model(input_shape=cnn_input_size, using_sparse_categorical_crossentropy=self.using_sparse_categorical_crossentropy)
        
        training_generator = neural_net.get_generator(training_data[train], training_data_labels[train], mini_batch_size=self.batch_size, using_sparse_categorical_crossentropy=self.using_sparse_categorical_crossentropy)
        validation_generator = neural_net.get_generator(training_data[test], training_data_labels[test], mini_batch_size=self.batch_size, using_sparse_categorical_crossentropy=self.using_sparse_categorical_crossentropy)
            
        model_save_name = self.save_name + str(j) + ".h5"
    
        # Callback methods
        checkpoint = ModelCheckpoint(model_save_name, monitor='loss', verbose=1, save_best_only=False, mode='min', period=100)
        logger = LossHistory()
        decrease_learning_rate_callback = MonitorStopping(model)

        callbacks = [checkpoint, logger, decrease_learning_rate_callback]
        train_net(model, training_generator, validation_generator, n_epochs, callbacks)
        
        logs_save_name = self.save_name + "_logs" + str(j)
        helper.save(model_save_name, logs_save_name, logger, model)
        j += 1

def train_without_crossvalidation(neural_net, cnn_input_size, using_sparse_categorical_crossentropy = False, validation_data_location="", validation_labels_location=""):
    model = neural_net.build_model(input_shape=cnn_input_size, using_sparse_categorical_crossentropy=using_sparse_categorical_crossentropy)
    training_generator = neural_net.get_generator(self.training_data, self.training_data_labels, mini_batch_size=self.batch_size, using_sparse_categorical_crossentropy=self.using_sparse_categorical_crossentropy)
        
    if(validation_data_location != ""):
        validation_d = helper.load_files([validation_data_location])
        validation_l = helper.load_files([validation_labels_location])
        # ohhhh, maybe patchcreater in model?
        validation_data, validation_data_labels = helper.patchCreator(validation_d, validation_l)
        validation_generator = neural_net.get_generator(validation_data, validation_data_labels, mini_batch_size=batch_size, using_sparse_categorical_crossentropy=using_sparse_categorical_crossentropy)
        
    model_save_name = save_name + ".h5"
    
    # Callback methods
    checkpoint = ModelCheckpoint(model_save_name, monitor='loss', verbose=1, save_best_only=False, mode='min', period=100)
    logger = LossHistory()
    decrease_learning_rate_callback = MonitorStopping(model)

    callbacks = [checkpoint, logger, decrease_learning_rate_callback]
        
    #def train(model, training_generator, n_epochs, callbacks, using_sparse_categorical_crossentropy=False):
    if(validation_data_location != ""):
        train_net(model, training_generator, validation_generator, self.n_epochs, callbacks, self.using_sparse_categorical_crossentropy, uses_validation_data=(validation_data_location != ""))
    else:
        train_net(model, training_generator, None, self.n_epochs, callbacks, self.using_sparse_categorical_crossentropy, uses_validation_data=(validation_data_location != ""))
        
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
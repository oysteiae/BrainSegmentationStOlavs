from numpy.random import seed
#from sklearn.cross_validation import KFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
import helper
import numpy as np

# TODO: using sparse_catecorical_entropy should maybe be called
# TODO: maybe use pickle to save the list used for crossvalidation.
# using_one_hot_encoding or something.
def train_crossvalidation(neural_net, training_data, training_labels, n_epochs, save_name, batch_size=4):
    j = 1
    seed = 7
    #kfold = KFold(n=len(training_data),n_folds=2, random_state=seed, shuffle=True)
    #for train, test in kfold:
    #    print("Testing on", helper.list_to_string(train))    
    #    print("Testing on", helper.list_to_string(test))
    
    #    model = None
    #    # Don't really need using_sparse_categorical_crossentropy
    #    model = neural_net.build_model()
        
    #    training_generator = neural_net.get_generator(training_data[train], training_labels[train], mini_batch_size=batch_size)
    #    validation_generator = neural_net.get_generator(training_data[test], training_labels[test], mini_batch_size=batch_size)
            
    #    model_save_name = save_name + str(j) + ".h5"

    #    callbacks = neural_net.get_callbacks(model_save_name, model)
        
    #    train_net(model, training_generator, validation_generator, n_epochs, callbacks)
        
    #    logs_save_name = save_name + "_logs" + str(j)
    #    helper.save(model_save_name, logs_save_name, callbacks[0], model)
    #    j += 1

def train_without_crossvalidation(neural_net, d, l, n_epochs, save_name, batch_size=4, use_validation=False, training_with_slurm=False, validation_data_location=None, validation_labels_location=None):
    validation_data=None
    validation_labels=None
    training_data=None
    training_labels=None
    # self, d, l, n_epochs, save_name, use_validation, training_with_slurm
    if(use_validation):
        print("Training with validation")
        if(training_with_slurm==False):
            training_indices, validation_indices = helper.compute_train_validation_test(d, l, save_name, training_with_slurm=training_with_slurm)
            print("Loading training data")
            training_data, training_labels = helper.patchCreator(d[training_indices], l[training_indices])
            print("Loading validation data")
            validation_data, validation_labels = helper.patchCreator(d[validation_indices], l[validation_indices])
        else:
            print("training with slurm")
            training_indices, validation_indices = helper.compute_train_validation_test(d, l, save_name, training_with_slurm=training_with_slurm)
            print("Loading training data")
            training_data, training_labels = helper.load_data_and_labels(d[training_indices], l[training_indices])
            print("Loading validation data")
            validation_data, validation_labels = helper.load_data_and_labels(d[validation_indices], l[validation_indices])
    else:
        print("Training without crossvalidation")
        if(training_with_slurm==False):
            if(validation_data_location is not None and validation_labels_location is not None):
                print("Training with validation from other source")
                vd = np.asarray(helper.load_files(validation_data_location))
                vl = np.asarray(helper.load_files(validation_labels_location))
                validation_data, validation_labels = helper.patchCreator(vd, vl, normalize=True, save_name=save_name)

            training_data, training_labels = helper.patchCreator(d, l, normalize=True, save_name=save_name)
        else:
            if(validation_data_location is not None and validation_labels_location is not None):
                print("Training with validation from other source")
                vd = np.asarray(helper.load_files(validation_data_location))
                vl = np.asarray(helper.load_files(validation_labels_location))
                validation_data, validation_labels = helper.load_data_and_labels(vd, vl)
            
            training_data, training_labels = helper.load_data_and_labels(d, l)

    model = neural_net.build_model()
    training_generator = neural_net.get_generator(training_data, training_labels, mini_batch_size=batch_size)
    
    # Validation data should not be sent in as a string.
    if(use_validation or (validation_data_location is not None and validation_labels_location is not None)):
        print("Creating validation generator")
        validation_generator = neural_net.get_generator(validation_data, validation_labels, mini_batch_size=1)
        
    model_save_name = save_name + ".h5"
    
    callbacks = neural_net.get_callbacks(model_save_name, model, save_name)
        
    if(validation_data is not None):
        print("Training with validation")
        train_net(model, training_generator, validation_generator, n_epochs, callbacks)
    else:
        train_net(model, training_generator, None, n_epochs, callbacks)
        
    helper.save(save_name, callbacks[0], neural_net.model_for_saving_weights, neural_net.gpus, training_with_slurm)

def train_net(model, training_generator, validation_generator, n_epochs, callbacks):
    if(validation_generator is not None):
        print("Training with validation")
        model.fit_generator(generator=training_generator,
            validation_data = validation_generator,
            validation_steps = 1,
            steps_per_epoch=1,
            epochs=n_epochs,
            verbose=0,
            callbacks=callbacks)
    else:
        model.fit_generator(generator=training_generator,
            steps_per_epoch=1,
            epochs=n_epochs,
            verbose=0,
            callbacks=callbacks)
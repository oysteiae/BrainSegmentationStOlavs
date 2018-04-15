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

def train_without_crossvalidation(neural_net, training_data, training_labels, n_epochs, save_name, batch_size=4, validation_data=None, validation_labels=None):
    model = neural_net.build_model()
    training_generator = neural_net.get_generator(training_data, training_labels, mini_batch_size=batch_size)
    
    # Validation data should not be sent in as a string.
    if(validation_data is not None):
        validation_generator = neural_net.get_generator(validation_data, validation_labels, mini_batch_size=1)
        
    model_save_name = save_name + ".h5"
    
    callbacks = neural_net.get_callbacks(model_save_name, model)
        
    #def train(model, training_generator, n_epochs, callbacks,
    #using_sparse_categorical_crossentropy=False):
    if(validation_data is not None):
        train_net(model, training_generator, validation_generator, n_epochs, callbacks, neural_net.using_sparse_categorical_crossentropy)
    else:
        train_net(model, training_generator, None, n_epochs, callbacks, neural_net.using_sparse_categorical_crossentropy)
        
    logs_save_name = save_name + "_logs"
    helper.save(model_save_name, logs_save_name, callbacks[0], neural_net.model_for_saving_weights)

def train_net(model, training_generator, validation_generator, n_epochs, callbacks, using_sparse_categorical_crossentropy=False):
    if(using_sparse_categorical_crossentropy):
        model.fit_generator(generator=training_generator,
            validation_data = validation_generator,
            validation_steps = 1,
            steps_per_epoch= 1,#len(training_data)/batch_size,
            epochs=n_epochs,
            verbose=0,
            callbacks=callbacks)
    if(validation_generator is not None):
        model.fit_generator(generator=training_generator,
            validation_data = validation_generator,
            validation_steps = 1,
            steps_per_epoch=1,
            epochs=n_epochs,
            verbose=2,
            callbacks=callbacks)
    else:
        model.fit_generator(generator=training_generator,
            steps_per_epoch=1,
            epochs=n_epochs,
            verbose=0,
            callbacks=callbacks)
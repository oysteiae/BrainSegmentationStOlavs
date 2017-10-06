from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, UpSampling3D
from keras.engine import Input, Model
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras import backend as K
from keras.layers.merge import concatenate
from keras.utils import Sequence
from keras import losses
#from keras.utils.vis_utils import plot_model
#import file_reading
import pickle
from random import shuffle
import nibabel as nib
import numpy as np
import os
from os import listdir as _listdir
from os.path import isfile as _isfile,join as  _join
import h5py

# TODO: rewrite this to something understandables.  Get rid of the current
def load_files(data_file_location=["C:\\Users\\oyste\\OneDrive\\MRI_SCANS\\data"], labels_file_location=["C:\\Users\\oyste\\OneDrive\\MRI_SCANS\\labels"]):
    data = []
    
    # TODO: Rewrite and remove these
    startswith = None
    endswith = None
    contains = None
    contains_not = None
    
    for path in data_file_location:
        gg = [ (_join(path,f) if path != "." else f) for f in _listdir(path) if _isfile(_join(path,f)) and (startswith == None or f.startswith(startswith)) and (endswith == None or f.endswith(endswith)) and (contains == None or contains in f) and (contains_not == None or (not (contains_not in f))) ]
        data+=gg

    labels = []
    for path in labels_file_location:
        gg = [ (_join(path,f) if path != "." else f) for f in _listdir(path) if _isfile(_join(path,f)) and (startswith == None or f.startswith(startswith)) and (endswith == None or f.endswith(endswith)) and (contains == None or contains in f) and (contains_not == None or (not (contains_not in f))) ]
        labels+=gg

    return data, labels

# TODO: rename
def load_file_as_nib(filename):
        return nib.load(filename).get_data()

# TODO rewrite so that you can set the parameters
# TODO maybe move to a class
def buildCNN(input_shape, pool_size=(2, 2, 2),
                  initial_learning_rate=0.00001, deconvolution=False, stride=1):
    inputs = Input(input_shape)
    conv1 = Conv3D(16, (4, 4, 4), strides=stride, activation='relu', padding='valid')(inputs)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(24, (5, 5, 5), strides=stride, activation='relu', padding='valid')(pool1)
    conv3 = Conv3D(28, (5, 5, 5), strides=stride, activation='relu', padding='valid')(conv2)
    conv4 = Conv3D(34, (5, 5, 5), strides=stride, activation='relu', padding='valid')(conv3)
    conv5 = Conv3D(42, (5, 5, 5), strides=stride, activation='relu', padding='valid')(conv4)
    conv6 = Conv3D(50, (5, 5, 5), strides=stride, activation='relu', padding='valid')(conv5)
    conv7 = Conv3D(50, (5, 5, 5), strides=stride, activation='relu', padding='valid')(conv6)

    #TODO the first argument should really be 2, I think
    conv8 = Conv3D(2, (1, 1, 1))(conv7)
    act = Activation('softmax')(conv8)
    model = Model(inputs=inputs, outputs=act)

    print(model.summary())
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# TODO: max fragment pooling
# TODO: Look at data Augmentation
# TODO: Find out what they mean with channels
def patchCreator(data, labels):
    files = zip(data, labels)
    q = []
    w = []
    for f in files:
        d = load_file_as_nib(f[0])
        
        # Removes the single dimensional entries in the array
        #d = np.squeeze(d)

        # They reshape the data to do the std and mean computation.
        # TODO: understand this a bit more
        # (176L, 208L, 176L, 1L)
        d2 = np.transpose(d,axes=[3,0,1,2])
        # (1L, 176L, 208L, 176L)
        d2 = np.reshape(d2,(d2.shape[0],-1))
        # (1L, 6443008L)
        std_ = np.std(d2,axis=1)
        mean_ = np.mean(d2,axis=1)
        # TODO: Why this calculation
        d = (d - mean_) / (4. * std_)
        q.append(d)

        l = load_file_as_nib(f[1])
        #Why don't they need the channel here?
        l = np.squeeze(l)
        l = (l > 0).astype('int16')
        w.append(l)

    return q,w

# def get_generator(data, labels, mini_batch_size=4):
# TODO: maybe add augmentation in the long run
def get_generator(data, labels, mini_batch_size=4):
    while True:
        x_list = list()
        y_list = list()
        
        for index in labels:
            #(data, labels, i_min, i_max, input_size, number_of_labeled_points_per_dim=4, stride=2, labels_offset=[26, 26, 26]
            dat, lab = get_cubes(data, labels, 0, len(data), 59)

            # TODO: do you need these extra dims?
            dat = np.expand_dims(dat, axis=0)
            lab = np.expand_dims(lab, axis=0)

            lab = np.expand_dims(lab, axis=4)
            yield (dat, lab)

""" picks <num> many cubes from [i_min,i_max)  (max is excluded) <num> many pictures."""
def get_cubes(data, labels, i_min, i_max, input_size, number_of_labeled_points_per_dim=4, stride=1, labels_offset=(26, 26, 26)):
    i = np.random.randint(i_min, i_max)# Used for selecting a random example
    dat = np.zeros( (input_size, input_size, input_size, 1), dtype="float32")
    labshape = (number_of_labeled_points_per_dim,)*3 #ndim
    
    lab = np.zeros( labshape, dtype="int16")
    #SP: (176L, 1L, 208L, 176L)
    #    [9, 141, 80]
    sp = data[0].shape

    off = [np.random.randint(0, sp[x] - input_size) for x in range(3)]
    dat = data[i][off[0] : off[0] + input_size, off[1] : off[1] + input_size, off[2] : off[2] + input_size, :]
    
    # TODO: these should be computed
    #number_of_labeled_points_per_dim:  4
    #CNET_stride 2
    loff = tuple(off) + labels_offset
    # Don't understand how this exactly works
    lab = labels[i] [loff[0] : loff[0] + number_of_labeled_points_per_dim * stride : stride, loff[1] : loff[1] + number_of_labeled_points_per_dim * stride : stride, loff[2]:loff[2] + number_of_labeled_points_per_dim * stride : stride]

    # Returns cubes of the training data
    return dat, lab

def run_on_slice(model, DATA):
    n_classes = 2
    CNET_stride = [2, 2, 2]
    pred_size = np.array([16, 16, 16])
    ImgInputSize = np.array([83, 83, 83])
    #(84, 84, 84, 1)
    #DATA = np.transpose(DATA,(0,3,1,2))
    DATA = DATA.reshape((1,) + DATA.shape)
    #(1, 83, 1, 83, 83)
    #(None, 83, 83, 83, 1)
     
    # TODO reimplement the stride stuff

    pred = np.zeros( (n_classes, ) + tuple(CNET_stride*pred_size),dtype=np.float32) # output
    rr = model.predict(DATA)

    return rr

def run_on_block(model, DATA):
    # TODO: calculate these yourself
    n_runs_p_dim = [6, 7, 6]
    ret_size_per_runonslice = 16
    n_classes = 2
    input_s = 83
    target_labels_per_dim = DATA.shape[:3]

    ret_3d_cube = np.zeros( tuple(   DATA.shape[:3]   ) , dtype="float32")
    for i in range(n_runs_p_dim[0]):
        print("COMPLETION =", 100.*i/n_runs_p_dim[0],"%")
        for j in range(n_runs_p_dim[1]):
            for k in range(n_runs_p_dim[2]): 
                offset = (ret_size_per_runonslice*i,ret_size_per_runonslice*(j),ret_size_per_runonslice*k)
                if DATA.ndim==4:
                    daa = DATA[offset[0]:input_s+offset[0],offset[1]:input_s+offset[1],offset[2]:input_s+offset[2],:]
                else:
                    daa = DATA[offset[0]:input_s+offset[0],offset[1]:input_s+offset[1],offset[2]:input_s+offset[2]]
                ret = run_on_slice(model, daa) 
                ret = ret.reshape(ret.shape[1:])
                ret = np.transpose(ret,(3,0,1,2))

                ret_3d_cube[offset[0]:ret_size_per_runonslice+offset[0], offset[1]:ret_size_per_runonslice+offset[1], offset[2]:ret_size_per_runonslice+offset[2]] = ret[0]
    sav = ret_3d_cube[:target_labels_per_dim[0],:target_labels_per_dim[1],:target_labels_per_dim[2]]
    print(sav.shape)
    sav = sav # pick class 1
    
    return sav

def predict():
    input_size = (83, 83, 83, 1)
    model = buildCNN(input_size)
    model.load_weights("model2.h5")
        
    d, l = load_files(data_file_location=["C:\\Users\\oyste\\OneDrive\\MRI_SCANS\\predict"])
    data, labels = patchCreator(d, l)
    predicted = run_on_block(model, data[0])
    
    nin = nib.Nifti1Image(predicted, None, None)
    nin.to_filename("test1")

def train_net():
    #Parameters
    initial_learning_rate = 0.00001
    learning_rate_drop = 0.5
    learning_rate_epochs = 100
    n_epochs = 1000
    steps_per_epoch = 10
    validation_split = 0.8
    
    input_size = (59, 59, 59, 1)

    d, l = load_files()
    data, labels = patchCreator(d, l)
    model = buildCNN(input_shape=input_size)
    
    n_training = int(len(data) * 0.8)
    training_data = data[:n_training]
    training_data_labels = labels[:n_training]
    validation_data = data[n_training:]
    validation_data_labels = labels[n_training:]
    
    training_generator = get_generator(training_data, training_data_labels)
    validation_generator = get_generator(validation_data, validation_data_labels)
   
    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_data),
        pickle_safe=False,
        verbose=2)
    
    model.save_weights("model2.h5")
    print("Saved model to disk")

# TODO: filter sizes
# TODO: I think the stride is 2, but I can't do anything because then the model doesn't fit.
# TODO: I think they use padding
# https://stackoverflow.com/questions/42945509/keras-input-shape-valueerror for theano shaping of the matrices
def main():
    predict()

main()
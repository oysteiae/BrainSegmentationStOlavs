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
import scipy.ndimage as ndimage

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
            #(data, labels, i_min, i_max, input_size,
            #number_of_labeled_points_per_dim=4, stride=2, labels_offset=[26,
            #26, 26]
            dat, lab = get_cubes(data, labels, 0, len(data), 59)
            
            yield (dat, lab)

# TODO: should compute number_of_labeled_points_per_dim myself
def get_cubes(data, labels, i_min, i_max, input_size, number_of_labeled_points_per_dim=4, stride=2):
    labels_offset = np.array((26, 26, 26))

    i = np.random.randint(i_min, i_max) # Used for selecting a random example
    dat = np.zeros((input_size, input_size, input_size, 1), dtype="float32")
    labshape = (number_of_labeled_points_per_dim,) * 3 #ndim
    
    lab = np.zeros(labshape, dtype="int16")
    data_shape = data[0].shape #shape = (176, 208, 176, 1)

    off = [np.random.randint(0, data_shape[x] - input_size) for x in range(3)]
    loff = tuple(off) + labels_offset #shape = (88, 146, 67)
    
    dat = data[i][off[0] : off[0] + input_size, off[1] : off[1] + input_size, off[2] : off[2] + input_size, :] #shape = (59, 59, 59, 1)
    lab = labels[i][loff[0] : loff[0] + number_of_labeled_points_per_dim * stride : stride, loff[1] : loff[1] + number_of_labeled_points_per_dim * stride : stride, loff[2]:loff[2] + number_of_labeled_points_per_dim * stride : stride] #shape = (4, 4, 4)
    
    # TODO: do you need these extra dims?
    dat = np.expand_dims(dat, axis=0) #shape = (1, 59, 59, 59, 1)
    lab = np.expand_dims(lab, axis=0)
    lab = np.expand_dims(lab, axis=4) #shape = (1, 4, 4, 4, 1)

    # Returns cubes of the training data
    return dat, lab

# Maybe you don't need this.
# TODO: find out exactly what this does.
def data_augmentation_greyvalue(data, max_shift=0.05, max_scale=1.3, min_scale=0.85, b_use_lesser_augmentation=0):
    if b_use_lesser_augmentation:
        max_shift = 0.02
        max_scale = 1.1
        min_scale = 0.91

    sh = (0.5 - np.random.random()) * max_shift * 2.
    scale = (max_scale - min_scale) * np.random.random() + min_scale
    return (sh + data * scale).astype("float32")

def greyvalue_data_padding(DATA, offset_l, offset_r):
    avg_value = 1. / 6. * (np.mean(DATA[0]) + np.mean(DATA[:,0]) + np.mean(DATA[:,:,0]) + np.mean(DATA[-1]) + np.mean(DATA[:,-1]) + np.mean(DATA[:,:,-1]))
    sp = DATA.shape
    
    #dat = avg_value * np.ones( (sp[0]+offset_l+offset_r if 0 in axis else
    #sp[0], sp[1]+offset_l+offset_r if 1 in axis else sp[1],
    #sp[2]+offset_l+offset_r if 2 in axis else sp[2]) + tuple(sp[3:]),
    #dtype="float32")
    dat = avg_value * np.ones((sp[0] + offset_l + offset_r, sp[1] + offset_l + offset_r, sp[2] + offset_l + offset_r) + tuple(sp[3:]), dtype="float32")
    #dat[offset_l*(0 in axis):offset_l*(0 in axis)+sp[0], offset_l*(1 in
    #axis):offset_l*(1 in axis)+sp[1], offset_l*(2 in axis):offset_l*(2 in
    #axis)+sp[2]] = DATA.copy()
    dat[offset_l : offset_l + sp[0], offset_l : offset_l + sp[1], offset_l : offset_l + sp[2]] = DATA.copy()
    
    return dat


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
    pred = np.zeros((n_classes,) + tuple(CNET_stride * pred_size),dtype=np.float32) # shape = (2, 32, 32, 32)
    for x in range(CNET_stride[0]):
        for y in range(CNET_stride[1]):
            for z in range(CNET_stride[2]):
                #rr = model.predict(DATA[:, x:x+ImgInputSize[0], :,
                #y:y+ImgInputSize[1], z:z+ImgInputSize[2]])
                rr = model.predict(DATA)
                #(1, 16, 16, 16, 2)
                # Filling every second element in pred starting from x, y, z
                pred[0, x::CNET_stride[0], y::CNET_stride[1], z::CNET_stride[2]] = rr[:,:,:,:, 0].reshape((pred_size[0], pred_size[1], pred_size[2])) # shape = (16, 16, 16)
    
    return pred

def run_on_block(model, DATA, rescale_predictions_to_max_range=True):
    # TODO: calculate these yourself
    n_runs_p_dim = [6, 7, 6]
    ret_size_per_runonslice = 32
                                        
    n_classes = 2
    input_s = 84
    target_labels_per_dim = DATA.shape[:3]
    print(DATA.shape)

    #offset_l = patchCreator.CNET_labels_offset[0]
    #offset_r = offset_l + input_s
    offset_l = 26
    offset_r = 110

    DATA = greyvalue_data_padding(DATA, offset_l, offset_r)

    ret_3d_cube = np.zeros(tuple(DATA.shape[:3]) , dtype="float32") # shape = (312, 344, 312)
    for i in range(n_runs_p_dim[0]):
        print("COMPLETION =", 100. * i / n_runs_p_dim[0],"%")
        for j in range(n_runs_p_dim[1]):
            for k in range(n_runs_p_dim[2]): 
                offset = (ret_size_per_runonslice * i, ret_size_per_runonslice * (j), ret_size_per_runonslice * k)
                #daa = DATA[offset[0] : input_s + offset[0], offset[1] :
                #input_s + offset[1], offset[2] : input_s + offset[2], :]
                daa = DATA[offset[0] : input_s + offset[0], offset[1] :  input_s + offset[1], offset[2] : input_s + offset[2], :]
                ret = run_on_slice(model, daa) 

                ret_3d_cube[offset[0] : ret_size_per_runonslice + offset[0], offset[1] : ret_size_per_runonslice + offset[1], offset[2] : ret_size_per_runonslice + offset[2]] = ret[0]
    
    # Hva gjør denne
    sav = ret_3d_cube[: target_labels_per_dim[0], : target_labels_per_dim[1], :target_labels_per_dim[2]]

    if rescale_predictions_to_max_range:
        sav = (sav-sav.min())/(sav.max()+1e-7) 
    
    return sav

def remove_small_conneceted_components(raw):
    """
    All but the two largest connected components will be removed
    """
    data = raw.copy()
    # binarize image
    data[data>0.5] = 1
    cc, num_components = ndimage.label(np.uint8(data))
    cc = cc.astype("uint16")
    # np.bincount computes how many instances there are of eaach value
    vals = np.bincount(cc.ravel())
    sizes = list(vals)
    try:
        second_largest = sorted(sizes)[::-1][1]       
    except:
        return raw.copy()
    
    data[...] = 0
    for i in range(0,len(vals)):
        # 0 is background
        if sizes[i]>=second_largest:
            data[cc==i] = raw[cc==i]
    return data

def predict(apply_cc_filtering=True):
    input_size = (84, 84, 84, 1)
    #input_size = (59, 59, 59, 1)
    save_name = "n_epochs_1000steps_per_epoch_100"
    model = buildCNN(input_size)
    model.load_weights(save_name + ".h5")
    
    d, l = load_files(data_file_location=["C:\\Users\\oyste\\OneDrive\\MRI_SCANS\\predict"])
    data, labels = patchCreator(d, l)
    predicted = run_on_block(model, data[0])
    
    if apply_cc_filtering:
        predicted = remove_small_conneceted_components(predicted)
        predicted = 1 - remove_small_conneceted_components(1 - predicted)
    
    nin = nib.Nifti1Image(predicted, None, None)
    nin.to_filename(save_name)

def train_net():
    #Parameters
    initial_learning_rate = 0.00001
    learning_rate_drop = 0.5
    learning_rate_epochs = 100
    n_epochs = 1000
    steps_per_epoch = 100
    validation_split = 0.8
    
    # TODO: determine input shape based on what you're training on.
    input_size = (59, 59, 59, 1)

    # Loads the files
    d, l = load_files()
    data, labels = patchCreator(d, l)
    model = buildCNN(input_shape=input_size)
    
    # Splits the data in to validation and training sets.
    n_training = int(len(data) * 0.8)
    training_data = data[:n_training]
    training_data_labels = labels[:n_training]
    validation_data = data[n_training:]
    validation_data_labels = labels[n_training:]
    
    training_generator = get_generator(training_data, training_data_labels)
    validation_generator = get_generator(validation_data, validation_data_labels)
   
    save_name = "n_epochs_" + str(n_epochs) + "steps_per_epoch_" + str(steps_per_epoch) + ".h5"
    model.fit_generator(generator=training_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_data),
        pickle_safe=False,
        verbose=2)
    
    model.save_weights(save_name)
    print("Saved model to disk")

# TODO: filter sizes
# TODO: I think the stride is 2, but I can't do anything because then the model
# doesn't fit.
# TODO: I think they use padding
# https://stackoverflow.com/questions/42945509/keras-input-shape-valueerror for
# theano shaping of the matrices
# TODO: Maybe you should randomize weights.
# TODO: Skriv en metode som kalkulerer hvor mange prosent som er feil.
# TODO: Test på andre bilder
def main():
    # HUSK å se på hvordan den lærer data som er i "midten"
    # De forstørrer dataen på en måte
    # Må ha greyvalue padding
    # Hva er forskjellen på greyvalue_pad_data og grey_value_data_padding
    #train_net()
    predict()
main()

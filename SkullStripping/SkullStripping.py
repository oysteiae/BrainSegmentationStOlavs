
from keras.layers import Activation
from keras.engine import Input, Model
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam, SGD
from keras.layers.convolutional import Conv3D, MaxPooling3D
import keras.backend as K
import nibabel as nib
import numpy as np
from numpy import random
from os import listdir as _listdir
from os.path import isfile as _isfile,join as  _join
import scipy.ndimage as ndimage
import itertools as it

# TODO: rewrite this to something understandables.  Get rid of the current
def load_files(data_file_location):
    data = []
    
    # TODO: Rewrite and remove these
    startswith = None
    endswith = None
    contains = None
    contains_not = None
    
    for path in data_file_location:
        gg = [ (_join(path,f) if path != "." else f) for f in _listdir(path) if _isfile(_join(path,f)) and (startswith == None or f.startswith(startswith)) and (endswith == None or f.endswith(endswith)) and (contains == None or contains in f) and (contains_not == None or (not (contains_not in f))) ]
        data+=gg

    return data

# TODO: rename
def load_file_as_nib(filename):
        return nib.load(filename).get_data()

# TODO rewrite so that you can set the parameters
# TODO maybe move to a class
def buildCNN(input_shape, pool_size=(2, 2, 2),
                  initial_learning_rate=0.00001, deconvolution=False, stride=1, using_sparse_categorical_crossentropy=False):
    inputs = Input(input_shape)
    conv1 = Conv3D(filters=16, kernel_size=(4, 4, 4), strides=stride, activation='relu', padding='valid')(inputs)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(filters=24, kernel_size=(5, 5, 5), strides=stride, activation='relu', padding='valid')(pool1)
    conv3 = Conv3D(filters=28, kernel_size=(5, 5, 5), strides=stride, activation='relu', padding='valid')(conv2)
    conv4 = Conv3D(filters=34, kernel_size=(5, 5, 5), strides=stride, activation='relu', padding='valid')(conv3)
    conv5 = Conv3D(filters=42, kernel_size=(5, 5, 5), strides=stride, activation='relu', padding='valid')(conv4)
    conv6 = Conv3D(filters=50, kernel_size=(5, 5, 5), strides=stride, activation='relu', padding='valid')(conv5)
    conv7 = Conv3D(filters=50, kernel_size=(5, 5, 5), strides=stride, activation='relu', padding='valid')(conv6)

    #TODO the first argument should really be 2, I think
    conv8 = Conv3D(filters=2, kernel_size=(1, 1, 1))(conv7)
    act = Activation('softmax')(conv8)
    model = Model(inputs=inputs, outputs=act)

    print(model.summary())
    if using_sparse_categorical_crossentropy:
        print("Using sparse categorical crossentropy as loss function")
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        # The loss function should perhaps be Kullback-Leibler divergence
        sgd = SGD(lr=initial_learning_rate)
        print("Using Kullback-Leibler divergence as loss function")
        #sgd = Adam(lr=initial_learning_rate)
        model.compile(optimizer=sgd, loss='kld', metrics=['accuracy'])
    
    return model

# TODO: max fragment pooling
# TODO: Look at data Augmentation
# TODO: Find out what they mean with channels
def patchCreator(data, labels):
    files = zip(data, labels)
    q = []
    w = []
    for f in files:
        f_split = f[0].split('.')

        if(f_split[-1] != "img"):
            print("Loading data:", f[0])
            print("Loading label:", f[1])
            d = load_file_as_nib(f[0])
            # If data doesn't have a channel we have to add it.
            if(d.ndim == 3):
                d = np.expand_dims(d, -1)

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
            # Maybe you can reverse this.
            l = (l > 0).astype('int16')
            w.append(l)

    return q,w

# def get_generator(data, labels, mini_batch_size=4):
# TODO: maybe add augmentation in the long run
# What does even mini_batch_size do here if you set it in the model.fit()?
def get_generator(data, labels, mini_batch_size=4, using_sparse_categorical_crossentropy=False):
    while True:
        x_list = list()
        y_list = list()
        
        #(data, labels, i_min, i_max, input_size,
        #number_of_labeled_points_per_dim=4, stride=2, labels_offset=[26, 26, 26]
        dat, lab = get_cubes(data, labels, 0, len(data), 59, using_sparse_categorical_crossentropy=using_sparse_categorical_crossentropy)
        dat = data_augmentation_greyvalue(dat)
             
        yield (dat, lab)

# TODO: should compute number_of_labeled_points_per_dim myself
# It's computed: ((Input_voxels_shape - 53)/2) + 1
# Should mulitply and add some numbers like they do in the article.
# The problems with the shapes: https://github.com/fchollet/keras/issues/4781
def get_cubes(data, labels, i_min, i_max, input_size, number_of_labeled_points_per_dim=4, stride=2, using_sparse_categorical_crossentropy=False):
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

# Maybe you don't need this.
# TODO: find out exactly what this does.
def data_augmentation_greyvalue(data, max_shift=0.05, max_scale=1.3, min_scale=0.85):
    sh = (0.5 - np.random.random()) * max_shift * 2.
    scale = (max_scale - min_scale) * np.random.random() + min_scale
    
    return (sh + data * scale).astype("float32")

def greyvalue_data_padding(DATA, offset_l, offset_r):
    avg_value = 1. / 6. * (np.mean(DATA[0]) + np.mean(DATA[:,0]) + np.mean(DATA[:,:,0]) + np.mean(DATA[-1]) + np.mean(DATA[:,-1]) + np.mean(DATA[:,:,-1]))
    sp = DATA.shape
    
    dat = avg_value * np.ones((sp[0] + offset_l + offset_r, sp[1] + offset_l + offset_r, sp[2] + offset_l + offset_r) + tuple(sp[3:]), dtype="float32")
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
    n_classes = 2
    input_s = 84
    target_labels_per_dim = DATA.shape[:3]
    
    ret_size_per_runonslice = 32
    n_runs_p_dim = [int(round(target_labels_per_dim[i] / ret_size_per_runonslice)) for i in [0,1,2]]
    print(DATA.shape)

    #offset_l = patchCreator.CNET_labels_offset[0]
    #offset_r = offset_l + input_s
    offset_l = 26
    offset_r = 110

    DATA = greyvalue_data_padding(DATA, offset_l, offset_r)

    ret_3d_cube = np.zeros(tuple(DATA.shape[:3]) , dtype="float32") # shape = (312, 344, 312)
    print(ret_3d_cube.shape)
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
    
    # What does this do
    sav = ret_3d_cube[: target_labels_per_dim[0], : target_labels_per_dim[1], :target_labels_per_dim[2]]
    if rescale_predictions_to_max_range:
        sav = (sav - sav.min()) / (sav.max() + 1e-7) 
    
    return sav

def remove_small_conneceted_components(raw):
    data = raw.copy()
    # binarize image
    data[data > 0.5] = 1
    cc, num_components = ndimage.label(np.uint8(data))
    cc = cc.astype("uint16")
    vals = np.bincount(cc.ravel())
    sizes = list(vals)
    try:
        second_largest = sorted(sizes)[::-1][1]       
    except:
        return raw.copy()
    
    data[...] = 0
    for i in range(0,len(vals)):
        # 0 is background
        if sizes[i] >= second_largest:
            data[cc == i] = raw[cc == i]
    return data

def predict(save_name, file_location=["C:\\Users\\oyste\\OneDrive\\MRI_SCANS\\predict"], apply_cc_filtering=True, using_sparse_categorical_crossentropy=False):
    cnn_input_size = (84, 84, 84, 1)
    #input_size = (59, 59, 59, 1)
    #save_name = "n_epochs_100_steps_per_epoch_100"
    model = buildCNN(cnn_input_size)
    model.load_weights(save_name + ".h5")
    #model = load_model(save_name + ".h5")

    d = load_files(file_location)
    #don't really need this but patchcreator needs to be rewritten to remove it
    l = load_files(file_location)
    
    print("Predicting file:", d[0])
    data, labels = patchCreator(d, l)
    sav = run_on_block(model, data[0])
    
    # TODO understand what this does
    if(apply_cc_filtering):
        predicted = remove_small_conneceted_components(sav)
        predicted = 1 - remove_small_conneceted_components(1 - sav)

    # Adding extra chanel so that it has equal shape as the input data.
    predicted = np.expand_dims(predicted, axis=4)

    nin = nib.Nifti1Image(predicted, None, None)
    nin.to_filename(d[0] + "_" + save_name + ".nii.gz")

    if(using_sparse_categorical_crossentropy):
        sav = (predicted <= 0.5).astype('int8')
    else:
        sav = (predicted > 0.5).astype('int8')

    nin = nib.Nifti1Image(sav, None, None)
    nin.to_filename(d[0] + "_" + save_name + "_masked.nii.gz")

def train_net(data_file_location=["C:\\Users\\oyste\\OneDrive\\MRI_SCANS\\data"], label_file_location=["C:\\Users\\oyste\\OneDrive\\MRI_SCANS\\labels"], using_sparse_categorical_crossentropy=False, load_model_name=""):
    #Parameters
    initial_learning_rate = 0.00001
    learning_rate_drop = 0.5
    learning_rate_epochs = 50
    n_epochs = 50000
    steps_per_epoch = 8
    validation_split = 0.8
    
    # TODO: determine input shape based on what you're training on.
    cnn_input_size = (59, 59, 59, 1)

    # Loads the files
    d = load_files(data_file_location)
    l = load_files(label_file_location)
    data, labels = patchCreator(d, l)
    
    if(load_model_name == ""):
        model = buildCNN(input_shape=cnn_input_size, using_sparse_categorical_crossentropy=using_sparse_categorical_crossentropy)
    else:
        model = load_model("scc.h5")
    
    # Splits the data in to validation and training sets.
    # validation data doesn't really do anything anymore does it?
    n_training = int(len(data) * 0.8)
    training_data = data[:n_training]
    training_data_labels = labels[:n_training]
    validation_data = data[n_training:]
    validation_data_labels = labels[n_training:]
    dat, lab = get_cubes(data, labels, 0, len(data), 59, using_sparse_categorical_crossentropy=using_sparse_categorical_crossentropy)

    training_generator = get_generator(training_data, training_data_labels, using_sparse_categorical_crossentropy=using_sparse_categorical_crossentropy)
    validation_generator = get_generator(validation_data, validation_data_labels, using_sparse_categorical_crossentropy=using_sparse_categorical_crossentropy)
   
    if using_sparse_categorical_crossentropy:
        save_name = "scc.h5"
    else:
        save_name = "kdr.h5"
    
    #This isn't completely configured yet.
    earlyStopping = EarlyStopping(monitor='val_loss',
                         min_delta=0,
                         patience=100, # You can experiment with this.
                         verbose=0, mode='auto')
    
    # Callback methods
    model_filepath = "C:\\Users\\oyste\\OneDrive\\MRI_SCANS\\predict\\" + save_name
    checkpoint = ModelCheckpoint(model_filepath, monitor='loss', verbose=1, save_best_only=False, mode='min', period=50)

    def scheduler(epoch):
        learning_rate = K.get_value(model.optimizer.lr)
        if (epoch % 5000) == 0 and epoch != 0:
            learning_rate *= 0.5
        return learning_rate
    decrease_learning_rate_callback = LearningRateScheduler(scheduler)

    if(using_sparse_categorical_crossentropy):
        model.fit_generator(generator=training_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=n_epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_data),
            pickle_safe=False,
            verbose=2,
            callbacks=[checkpoint])    
    else:
        #callbacks=[checkpoint, decrease_learning_rate_callback]
        model.fit_generator(generator=training_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=n_epochs,
            #validation_data=validation_generator,
            #validation_steps=len(validation_data),
            pickle_safe=False,
            verbose=2,
            callbacks=[checkpoint, decrease_learning_rate_callback])   
    
    #model.save_weights(save_name)
    model.save_weights(save_name)
    print("Saved model to disk")

def compute_scores(pred, label):
    # Pred and label must have the same shape
    shape = pred.shape
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(0, shape[0]):
        print("Comleted", float(i)/float(shape[0]) * 100, "%")
        for j in range(0, shape[1]):
            for k in range(0, shape[2]):
                if(pred[i][j][k] == 1 and label[i][j][k] >= 1):
                    TP += 1
                elif(pred[i][j][k] == 1 and label[i][j][k] == 0):
                    FP += 1
                elif(pred[i][j][k] == 0 and label[i][j][k] >= 1):
                    FN += 1  
                elif(pred[i][j][k] == 0 and label[i][j][k] == 0):
                    TN += 1

    dice_coefficient = (2 * TP) / (2 * TP + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    print("dice_coefficient:", dice_coefficient)
    print("sensitivity:", sensitivity)
    print("specificity:", specificity)

# TODO: filter sizes
# https://stackoverflow.com/questions/42945509/keras-input-shape-valueerror for
# theano shaping of the matrices
# TODO: Maybe you should randomize weights.
# TODO: Test på andre bilder
# TODO: Write your own function for finding the optimal input size.
# TODO: You can add error checking
# Find lots of training data
# Correct MRI scans for the "pollution" in code.
def main():
    # Hva er forskjellen på greyvalue_pad_data og grey_value_data_padding
    train_net(using_sparse_categorical_crossentropy=False)
    #predict("kdr_more_nperepochs", using_sparse_categorical_crossentropy=False)
    
    #pred = load_files(["C:\\Users\\oyste\\OneDrive\\MRI_SCANS\\own_predictions"])
    #gt = load_files(["C:\\Users\\oyste\\OneDrive\\MRI_SCANS\\gt"])
    #data = load_file_as_nib(pred[0])
    #label = load_file_as_nib(gt[0])
    #compute_scores(data, label)
main()
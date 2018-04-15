import nibabel as nib
import numpy as np
from os import listdir as _listdir, getcwd
from os.path import isfile as _isfile,join as  _join, abspath, splitext
from pathlib import Path
import ntpath
import pickle

# Taken from https://github.com/GUR9000/Deep_MRI_brain_extraction
def load_files(data_file_location):
    data = []
    
    startswith = None
    endswith = None
    contains = None
    contains_not = None
    
    for path in data_file_location:
        gg = [ (_join(path,f) if path != "." else f) for f in _listdir(path) if _isfile(_join(path,f)) and (startswith == None or f.startswith(startswith)) and (endswith == None or f.endswith(endswith)) and (contains == None or contains in f) and (contains_not == None or (not (contains_not in f))) ]
        data+=gg

    return data

def load_file_as_nib(filename):
    return nib.load(filename).get_data()

def process_labels(labels):
    w = []
    labels = sorted(labels)
    
    for label in labels:
        d_split = label.split('.')
        
        if(d_split[-1] != "img"):
            print(label)
            l = load_file_as_nib(label)
            l = np.squeeze(l)
            l = (l > 0).astype('int16')
            w.append(l)
    
    print("Finished loading labels")
    return np.asarray(w)

def process_data(data, normalize=True):
    q = []
    data = sorted(data)

    for da in data:
        d_split = da.split('.')

        if(d_split[-1] != "img"):
            print(da)
            d = load_file_as_nib(da)
            
            # If data doesn't have a channel we have to add it.
            if(d.ndim == 3):
                d = np.expand_dims(d, -1)

            # They reshape the data to do the std and mean computation.
            d2 = np.transpose(d,axes=[3,0,1,2])
            d2 = np.reshape(d2,(d2.shape[0],-1))
            std_ = np.std(d2,axis=1)
            mean_ = np.mean(d2,axis=1)

            d = (d - mean_) / (4. * std_)

            q.append(d)
    print("Finished loading data")
    return np.asarray(q)

def patchCreator(data, labels, normalize=True):
    return process_data(data, normalize), process_labels(labels)

def get_parent_directory():
    return str(Path(getcwd()).parent)

def save(save_name, log_save_name, logger, model):
    parentDirectory = get_parent_directory()
    model.save_weights(parentDirectory + "/models/" + save_name)
    #model.save_weights("/home/oysteiae/models/" + save_name)
    print("Saved model to disk")

    log_name = parentDirectory + "/logs/" + log_save_name + ".tsv"
    #log_name = "/home/oysteiae/logs/" + log_save_name + ".tsv"

    with open(log_name, "w") as logs:
        logs.write("Epoch\tAcc\tLoss\tTime\tvalloss\tvalacc\n")
        for i in range(len(logger.accuracies)):
            logs.write(str(i) + "\t" + str(logger.accuracies[i]) + "\t" + str(logger.losses[i]) + "\t" + str(logger.timestamp[i]) + "\t" + str(logger.val_losses[i]) + "\t" + str(logger.val_accuracies[i]) + "\n")
    
    print("Saved logs to disk")

# TODO: Maybe you want the header information here as well.
def save_prediction(save_name_extension, predicted, original_file_name, using_sparse_categorical_crossentropy=False):
    original_file_name_without_path = ntpath.basename(original_file_name).split('.')[0]
    parentDirectory = get_parent_directory()
    save_name = original_file_name_without_path + "_" + save_name_extension

    nin = nib.Nifti1Image(predicted, None, None)
    nin.to_filename(parentDirectory + "/predicted/" + save_name + ".nii.gz")
    
    if(using_sparse_categorical_crossentropy):
        sav = (predicted <= 0.5).astype('int8')
    else:
        sav = (predicted > 0.5).astype('int8')

    nin = nib.Nifti1Image(sav, None, None)
    nin.to_filename(parentDirectory + "/predicted/" + save_name + "_masked.nii.gz")

    print("Saved prediction", save_name, "to " +  parentDirectory + "/predicted/")

def list_to_string(list):
    string_list = ""
    for elem in list:
        string_list += str(elem) + ","
    
    return string_list

def compute_train_validation_test(data_files, label_files, save_name):
    data_list = np.copy(data_files)
    label_list = np.copy(label_files)

    indices = np.arange(0, len(data_list), dtype=int)
    np.random.shuffle(indices)

    training_part = 0.6
    validation_part = 0.2
    testing_part = 0.2
    training_len = (int)(len(data_list) * training_part)
    validation_len = (int)(len(data_list) * validation_part)
    testing_len = (int)(len(data_list) * testing_part)

    training_indices = indices[ : training_len]
    validation_indices = indices[training_len : validation_len + training_len]
    testing_indices = indices[training_len + validation_len : ]

    with open("training_indices" + save_name + ".txt", "wb") as tr:
        pickle.dump(training_indices, tr)
    with open("validation_indices" + save_name + ".txt", "wb") as va:
        pickle.dump(validation_indices, va)
    with open("testing_indices" + save_name + ".txt", "wb") as te:
        pickle.dump(testing_indices, te)
    
    return training_indices, validation_indices
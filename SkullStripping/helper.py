import nibabel as nib
import numpy as np
from os import listdir as _listdir, getcwd, mkdir, path
from os.path import isfile as _isfile,join as  _join, abspath, splitext
from pathlib import Path
import ntpath
import pickle
import inspect

# Taken from https://github.com/GUR9000/Deep_MRI_brain_extraction
def load_files(data_file_location):
    data = []
    
    startswith = None
    endswith = None
    contains = None
    contains_not = None
    
    for path in data_file_location:
        gg = [ (_join(path,f) if path != "." else f) for f in _listdir(path) if _isfile(_join(path,f)) and (startswith == None or f.startswith(startswith)) and (endswith == None or f.endswith(endswith)) and (contains == None or contains in f) and (contains_not == None or (not (contains_not in f))) ]
        data.append(gg)

    combined_list = []
    # Sort the lists:
    for i in range(len(data)):
        elem = sorted(data[i])
        combined_list = combined_list + elem
    
    combined_list = strip_files_of_doubles(combined_list)
    combined_list = sorted(combined_list, key=sort_func)
    return combined_list

def sort_func(s):
   sort_string = s.split('/')[-1].rstrip()
   return sort_string

def strip_files_of_doubles(files):
    new_files = []
    for file in files:
        split = file.split('.')
        if(split[-1] != "img" and split[-1] != "mat"):
            new_files.append(file)

    return new_files

def load_file_as_nib(filename):
    file =  np.asarray(nib.load(filename).dataobj)
    print(filename)
    return file

def process_labels(labels, save_name=""):
    w = []

    for label in labels:
        d_split = label.split('.')
        
        if(d_split[-1] != "img" and d_split[-1] != "mat"):
            l = load_file_as_nib(label)
            l = np.squeeze(l)
            l = (l > 0).astype('int16')

            #if(l.ndim == 3):
            #    l = np.expand_dims(l, -1)
            
            w.append(l)

    
    print("Finished loading labels")
    return w

def process_data(data, normalize=True, save_name=""):
    q = []

    for da in data:
        d_split = da.split('.')

        if(d_split[-1] != "img" and d_split[-1] != "mat"):
            d = load_file_as_nib(da)
            
            # If data doesn't have the last channel we have to add it.
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
    return q

def patchCreator(data, labels, normalize=True, save_name=""):
    return process_data(data, normalize, save_name), process_labels(labels, save_name)

def load_data_and_labels(data, labels):
    q = [load_file_as_nib(x) for x in data]
    w = [load_file_as_nib(x) for x in labels]

    return q, w

def get_parent_directory():
    return str(Path(getcwd()).parent)

def save(save_name, logger, model, gpus=1, training_with_slurm=False):
    parentDirectory = get_parent_directory()
    experiment_directory = parentDirectory + "/Experiments/" + save_name + "/"
    if(training_with_slurm == False):
        try:    
            mkdir(experiment_directory)
        except FileExistsError:
            print("Folder exists, do nothing")
    else:
        try:
            mkdir("/home/oysteiae/Experiments/" + save_name + "/")        
        except FileExistsError:
            print("Folder exists, do nothing")
    
    if(training_with_slurm==False):
        model.save_weights(experiment_directory + save_name + ".h5")
        log_name = experiment_directory + save_name + "_logs.tsv"
    else:
        model.save_weights("/home/oysteiae/Experiments/" + save_name + "/" + save_name + ".h5")
        log_name = "/home/oysteiae/Experiments/" + save_name + "/" + save_name + "_logs.tsv"
    
    print("Saved model to disk")

    #print(parentDirectory + "/" + save_name)
    with open(log_name, "w") as logs:
        logs.write("Epoch\tAcc\tLoss\tTime\tvalloss\tvalacc\n")
        for i in range(len(logger.accuracies)):
            logs.write(str(i) + "\t" + str(logger.accuracies[i]) + "\t" + str(logger.losses[i]) + "\t" + str(logger.timestamp[i]) + "\t" + str(logger.val_losses[i]) + "\t" + str(logger.val_accuracies[i]) + "\n")
    
    print("Saved logs to disk")

def save_prediction(save_name_extension, predicted, original_file_name, using_sparse_categorical_crossentropy=False, original_file=None):
    original_file_name_without_path = ntpath.basename(original_file_name).split('.')[0]
    parentDirectory = get_parent_directory()
    save_name = original_file_name_without_path + "_" + save_name_extension

    if(original_file is not None):
        nin = nib.Nifti1Image(predicted, None, nib.load(original_file).get_header())
    else:
        nin = nib.Nifti1Image(predicted, None, None)
    nin.to_filename(parentDirectory + "/predicted/" + save_name + ".nii.gz")
    
    if(using_sparse_categorical_crossentropy):
        sav = (predicted <= 0.5).astype('int8')
    else:
        sav = (predicted > 0.5).astype('int8')

    if(original_file is not None):
        nin = nib.Nifti1Image(sav, None, nib.load(original_file).get_header())
    else:
        nin = nib.Nifti1Image(sav, None, None)

    nin.to_filename(parentDirectory + "/predicted/" + save_name + "_masked.nii.gz")

    print("Saved prediction", save_name, "to " +  parentDirectory + "/predicted/")

def load_weights_for_experiment(model, model_save_name, evaluating_with_slurm):
    if(evaluating_with_slurm):
        parentDirectory = "/home/oysteiae/"
    else:
        parentDirectory = get_parent_directory()

    model.load_weights(parentDirectory + "/Experiments/" + model_save_name + "/" + model_save_name + ".h5")

def open_score_file(save_name, evaluating_with_slurm, part_to_test_on):
    if(evaluating_with_slurm):
        parentDirectory = "/home/oysteiae/"
    else:
        parentDirectory = get_parent_directory()
    if(part_to_test_on is not None):
        return open(parentDirectory + "/Experiments/" + save_name + "/" + part_to_test_on + "_" + save_name + "_scores.tsv", 'w')
    else:
        return open(parentDirectory + "/Experiments/" + save_name + "/" + save_name + "_scores.tsv", 'w')

def list_to_string(list):
    string_list = ""
    for elem in list:
        string_list += str(elem) + ","
    
    return string_list

def load_indices(save_name, indice_name, evaluating_with_slurm=False):
    if(evaluating_with_slurm):
        parentDirectory = "/home/oysteiae/"
    else:
        parentDirectory = get_parent_directory()
    with open(parentDirectory + "/Experiments/" + save_name + "/" + indice_name + save_name + ".txt", "rb") as fp:
        indices = pickle.load(fp)
    print(indice_name + save_name + ".txt")
    return indices

def compute_train_validation_test(data_files, label_files, save_name, training_with_slurm=False):
    data_list = np.copy(data_files)
    label_list = np.copy(label_files)
    print(save_name)

    if(training_with_slurm == False):
        parentDirectory = get_parent_directory()
        experiment_directory = parentDirectory + "/Experiments/" + save_name + "/"
    else:
        experiment_directory = "/home/oysteiae/Experiments/" + save_name + "/"

    path_to_training_file = experiment_directory +  "training_indices" + save_name + ".txt"
    path_to_validation_file = experiment_directory + "validation_indices" + save_name + ".txt"
        
    if(Path(path_to_training_file).is_file()):
        with open(path_to_training_file, "rb") as tp:
            training_indices = pickle.load(tp)
        with open(path_to_validation_file, "rb") as vp:
            validation_indices = pickle.load(vp)
        print("isfile")
        return training_indices, validation_indices

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

    try:
        mkdir(experiment_directory)
    except FileExistsError:
        print("Folder exists, do nothing")

    with open(experiment_directory + "training_indices" + save_name + ".txt", "wb") as tr:
        pickle.dump(training_indices, tr)
    with open(experiment_directory + "validation_indices" + save_name + ".txt", "wb") as va:
        pickle.dump(validation_indices, va)
    with open(experiment_directory + "testing_indices" + save_name + ".txt", "wb") as te:
        pickle.dump(testing_indices, te)
    
    return training_indices, validation_indices
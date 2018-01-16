import nibabel as nib
import numpy as np
from os import listdir as _listdir
from os.path import isfile as _isfile,join as  _join

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

def patchCreator(data, labels):
    files = zip(data, labels)
    q = []
    w = []
    
    for f in files:
        f_split = f[0].split('.')

        if(f_split[-1] != "img"):
            d = load_file_as_nib(f[0])
            
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

            l = load_file_as_nib(f[1])
            l = np.squeeze(l)
            l = (l > 0).astype('int16')
            w.append(l)

    #return np.asarray(q), np.asarray(w)
    return q, w
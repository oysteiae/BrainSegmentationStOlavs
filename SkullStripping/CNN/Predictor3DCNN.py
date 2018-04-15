import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage
import helper
import extra
from CNN.Build3DCNN import build_3DCNN

class Predictor3DCNN:
    'Class used for predicting MRI images with a 3D CNN'
    def __init__(self, save_name, gpus, apply_cc_filtering=True, using_sparse_categorical_crossentropy=False):
        self.input_size = (84, 84, 84, 1)
        self.save_name = save_name
        self.using_sparse_categorical_crossentropy = using_sparse_categorical_crossentropy
        self.apply_cc_filtering = apply_cc_filter
        
        self.model, parallel_model = build_3DCNN(self.input_size, gpus)
        self.model.load_weights(save_name + ".h5")

    def predict(self, file_location):
        d = helper.load_files(file_location)
        data = helper.process_data(d, True)

        for i in range(0, len(data)):
            print("Predicting file:", d[i])
            sav = self.predict_data(self.model, data[i], self.input_size)
    
            # TODO understand what this does
            if(self.apply_cc_filtering):
                predicted = self.remove_small_conneceted_components(sav)
                predicted = 1 - self.remove_small_conneceted_components(1 - sav)

            # Adding extra chanel so that it has equal shape as the input data.
            predicted = np.expand_dims(predicted, axis=4)

            helper.save_prediction(self.save_name, predicted, d[i], self.using_sparse_categorical_crossentropy)

    def predict_data(self, model, DATA, input_size, rescale_predictions_to_max_range=True):
        n_classes = 2
        input_s = 84
        target_labels_per_dim = DATA.shape[:3]
    
        ret_size_per_runonslice = 32
        n_runs_p_dim = [int(round(target_labels_per_dim[i] / ret_size_per_runonslice)) for i in [0,1,2]]

        # TODO: Set or calculate these numbers yourself.
        offset_l = 26
        offset_r = 110

        DATA = extra.greyvalue_data_padding(DATA, offset_l, offset_r)

        ret_3d_cube = np.zeros(tuple(DATA.shape[:3]) , dtype="float32") # shape = (312, 344, 312)
        for i in range(n_runs_p_dim[0]):
            print("COMPLETION =", 100. * i / n_runs_p_dim[0],"%")
            for j in range(n_runs_p_dim[1]):
                for k in range(n_runs_p_dim[2]): 
                    offset = (ret_size_per_runonslice * i, ret_size_per_runonslice * (j), ret_size_per_runonslice * k)
                    daa = DATA[offset[0] : input_s + offset[0], offset[1] :  input_s + offset[1], offset[2] : input_s + offset[2], :]
                    ret = self.run_on_slice(daa) 

                    ret_3d_cube[offset[0] : ret_size_per_runonslice + offset[0], offset[1] : ret_size_per_runonslice + offset[1], offset[2] : ret_size_per_runonslice + offset[2]] = ret[0]
    
        sav = ret_3d_cube[: target_labels_per_dim[0], : target_labels_per_dim[1], :target_labels_per_dim[2]]
        if rescale_predictions_to_max_range:
            sav = (sav - sav.min()) / (sav.max() + 1e-7) 
    
        return sav

    def run_on_slice(self, DATA):
        # TODO: Maybe define these for the class
        n_classes = 2
        CNET_stride = [2, 2, 2]
        pred_size = np.array([16, 16, 16])
        DATA = DATA.reshape((1,) + DATA.shape) 
     
        # TODO reimplement the stride stuff
        # Test without the stride stuf?f
        # Check of large the rr is, it should be (16, 16, 16)
        pred = np.zeros((n_classes,) + tuple(CNET_stride * pred_size),dtype=np.float32) # shape = (2, 32, 32, 32)
        rr = self.model.predict(DATA)
        for x in range(CNET_stride[0]):
            for y in range(CNET_stride[1]):
                for z in range(CNET_stride[2]):
                    pred[0, x::CNET_stride[0], y::CNET_stride[1], z::CNET_stride[2]] = rr[:,:,:,:, 0].reshape((pred_size[0], pred_size[1], pred_size[2])) # shape = (16, 16, 16)
    
        return pred

    # Taken from: https://github.com/GUR9000/Deep_MRI_brain_extraction
    def remove_small_connected_components(self, raw):
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
            if sizes[i] >= second_largest:
                data[cc == i] = raw[cc == i]
        return data
import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage
import helper

class Predictor3DCNN:
    'Class used for predicting MRI images with a 3D CNN'
    def __init__(self, save_name, file_location, build_CNN, apply_cc_filtering=True, using_sparse_categorical_crossentropy=False):
        self.cnn_input_size = (84, 84, 84, 1)
        #input_size = (59, 59, 59, 1)
        #save_name = "n_epochs_100_steps_per_epoch_100"
        self.model = build_CNN(self.cnn_input_size)
        self.model.load_weights(save_name + ".h5")
        #model = load_model(save_name + ".h5")

        self.d = helper.load_files(file_location)
        #don't really need this but patchcreator needs to be rewritten to remove it
        self.l = helper.load_files(file_location)
    
        self.data, self.labels = helper.patchCreator(self.d, self.l)

    def predict(self):
        for i in range(0, len(self.data)):
            print("Predicting file:", self.d[i])
            sav = self.run_on_block(self.data[i])
    
            # TODO understand what this does
            if(apply_cc_filtering):
                predicted = self.remove_small_conneceted_components(sav)
                predicted = 1 - self.remove_small_conneceted_components(1 - sav)

            # Adding extra chanel so that it has equal shape as the input data.
            predicted = np.expand_dims(predicted, axis=4)

            nin = nib.Nifti1Image(predicted, None, None)
            nin.to_filename(d[i] + "_" + save_name + ".nii.gz")

            if(using_sparse_categorical_crossentropy):
                sav = (predicted <= 0.5).astype('int8')
            else:
                sav = (predicted > 0.5).astype('int8')

            nin = nib.Nifti1Image(sav, None, None)
            nin.to_filename(d[i] + "_" + save_name + "_masked.nii.gz")

    def run_on_slice(self, DATA):
        # TODO: Maybe define these for the class
        n_classes = 2
        CNET_stride = [2, 2, 2]
        pred_size = np.array([16, 16, 16])
        ImgInputSize = np.array([83, 83, 83])
        DATA = DATA.reshape((1,) + DATA.shape) 
     
        # TODO reimplement the stride stuff
        pred = np.zeros((n_classes,) + tuple(CNET_stride * pred_size),dtype=np.float32) # shape = (2, 32, 32, 32)
        for x in range(CNET_stride[0]):
            for y in range(CNET_stride[1]):
                for z in range(CNET_stride[2]):
                    rr = self.model.predict(DATA)
                    pred[0, x::CNET_stride[0], y::CNET_stride[1], z::CNET_stride[2]] = rr[:,:,:,:, 0].reshape((pred_size[0], pred_size[1], pred_size[2])) # shape = (16, 16, 16)
    
        return pred

    def run_on_block(self, DATA, rescale_predictions_to_max_range=True):
        n_classes = 2
        input_s = 84
        target_labels_per_dim = DATA.shape[:3]
    
        ret_size_per_runonslice = 32
        n_runs_p_dim = [int(round(target_labels_per_dim[i] / ret_size_per_runonslice)) for i in [0,1,2]]

        # TODO: Set or calculate these numbers yourself.
        offset_l = 26
        offset_r = 110

        DATA = self.greyvalue_data_padding(DATA, offset_l, offset_r)

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

    # Taken from: https://github.com/GUR9000/Deep_MRI_brain_extraction
    def greyvalue_data_padding(self, DATA, offset_l, offset_r):
        avg_value = 1. / 6. * (np.mean(DATA[0]) + np.mean(DATA[:,0]) + np.mean(DATA[:,:,0]) + np.mean(DATA[-1]) + np.mean(DATA[:,-1]) + np.mean(DATA[:,:,-1]))
        sp = DATA.shape
    
        dat = avg_value * np.ones((sp[0] + offset_l + offset_r, sp[1] + offset_l + offset_r, sp[2] + offset_l + offset_r) + tuple(sp[3:]), dtype="float32")
        dat[offset_l : offset_l + sp[0], offset_l : offset_l + sp[1], offset_l : offset_l + sp[2]] = DATA.copy()
    
        return dat
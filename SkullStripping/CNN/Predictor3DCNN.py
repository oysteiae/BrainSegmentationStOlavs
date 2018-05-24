import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage
import helper
import extra
from CNN.Build3DCNN import build_3DCNN
from CNN.helper_methods_CNN import compute_label_offset
import ntpath

class Predictor3DCNN:
    'Class used for predicting MRI images with a 3D CNN'
    def __init__(self, save_name, gpus, apply_cc_filtering=True, evaluating_with_slurm=False, input_size=(84, 84, 84, 1), loss_function='kld', use_validation=False, part_to_test_on=None):
        self.input_size = input_size
        self.save_name = save_name
        self.apply_cc_filtering = apply_cc_filtering
        self.model, parallel_model = build_3DCNN(self.input_size, gpus, loss_function)
        self.output_size = self.model.layers[-1].output_shape
        print(self.output_size)
        self.CNET_stride = np.array((2, 2, 2), dtype='int16')

        self.use_validation=use_validation
        self.part_to_test_on = part_to_test_on

        helper.load_weights_for_experiment(self.model, save_name, evaluating_with_slurm)

    def predict(self, file_location):
        d = np.asarray(helper.load_files(file_location))
        if(self.use_validation and self.part_to_test_on is not None):
            data = helper.process_data(d[helper.load_indices(self.save_name, self.part_to_test_on, False)], True)
        else:
            data = helper.process_data(d, True)

        for i in range(0, len(data)):
            print("Predicting file:", d[i])
            sav = self.predict_data(self.model, data[i], self.input_size)
    
            helper.save_prediction(ntpath.basename(d[i]).split('.')[0], sav, self.save_name + "_pred_", original_file=d[i])

    def predict_data(self, model, DATA, input_size, rescale_predictions_to_max_range=True, stride=2):
        n_classes = 2
        input_s = input_size[0]
        target_labels_per_dim = DATA.shape[:3]
        
        ret_size_per_runonslice = self.output_size[1] * stride
        n_runs_p_dim = [int(round(target_labels_per_dim[i] / ret_size_per_runonslice)) for i in [0,1,2]]

        offset_l = compute_label_offset()[0]
        offset_r = offset_l + input_s

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
        
        if(self.apply_cc_filtering):
            predicted = self.remove_small_connected_components(sav)
            predicted = 1 - self.remove_small_connected_components(1 - sav)

        return predicted

    def run_on_slice(self, DATA):
        n_classes = self.output_size[-1]
        pred_size = self.output_size[1:4]
        DATA = DATA.reshape((1,) + DATA.shape) 
     
        pred = np.zeros((n_classes,) + tuple(self.CNET_stride * pred_size),dtype=np.float32)
        rr = self.model.predict(DATA)
        
        for x in range(self.CNET_stride[0]):
            for y in range(self.CNET_stride[1]):
                for z in range(self.CNET_stride[2]):
                    for a in range(n_classes):
                        # For now the class thing is useless since so much of the code depends on it only predicting two classes
                        pred[a, x::self.CNET_stride[0], y::self.CNET_stride[1], z::self.CNET_stride[2]] = rr[:,:,:,:, a].reshape((pred_size[0], pred_size[1], pred_size[2]))
    
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
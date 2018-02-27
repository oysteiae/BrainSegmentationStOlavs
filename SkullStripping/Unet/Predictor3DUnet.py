import nibabel as nib
import numpy as np
from Unet.Build3DUnet import build_3DUnet
import helper

class Predictor3DUnet:
    """description of class"""
    def __init__(self, save_name, file_location, input_size):
        self.d = helper.load_files(file_location)
        self.save_name = save_name
        self.data = helper.process_data(self.d)
        self.input_size = input_size

        self.unet = build_3DUnet(input_size)
        self.unet.load_weights(save_name + ".h5")

    def predict(self):
        for elem in self.data:
            pred = self.predict_block(elem)
            #nin = nib.Nifti1Image(sav, None, None)
            #nin.to_filename("D:\\MRI_SCANS\\" + self.save_name + "_masked.nii.gz")
            helper.save_prediction("unet", pred, "unet", False)

    # TODO: This needs implemented properly.
    def predict_block(self, DATA):
        target_labels_per_dim = DATA.shape[:3]
        ret_size_per_runonslice = 40
        n_runs_p_dim = [int(round(target_labels_per_dim[i] / ret_size_per_runonslice)) for i in [0,1,2]]
        ret_3d_cube = np.zeros(tuple(DATA.shape) , dtype="float32") # shape = (312, 344, 312)
        
        for i in range(n_runs_p_dim[0]):
            print("COMPLETION =", 100. * i / n_runs_p_dim[0],"%")
            for j in range(n_runs_p_dim[1]):
                for k in range(n_runs_p_dim[2]): 
                    offset = (ret_size_per_runonslice * i, ret_size_per_runonslice * (j), ret_size_per_runonslice * k)
                    
                    daa = DATA[offset[0] : ret_size_per_runonslice + offset[0], offset[1] :  ret_size_per_runonslice + offset[1], offset[2] : ret_size_per_runonslice + offset[2], :]
                    daa = np.expand_dims(daa, axis=0)
                    ret = self.unet.predict(daa) 

                    ret_3d_cube[offset[0] : ret_size_per_runonslice + offset[0], offset[1] : ret_size_per_runonslice + offset[1], offset[2] : ret_size_per_runonslice + offset[2]] = ret[0]
        
        return ret_3d_cube
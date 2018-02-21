import nibabel as nib
from Build3DUnet import build_3DUnet
import helper

class Predictor3DUnet:
    """description of class"""
    def __init__(self, save_name, file_location, input_size, initial_learning_rate=0.001):
        self.d = helper.load_files(file_location)
        self.save_same = save_name
        self.data = helper.process_data(self.d)
        self.input_size = input_size
        self.unet = build_3DUnet(input_shape, use_upsampling=False)
        self.unet.load_weights(save_name + ".h5")
        
    # TODO: This needs implemented properly.
    def predict(self):
        target_labels_per_dim = DATA.shape[:3]
        n_runs_p_dim = [int(round(target_labels_per_dim[i] / ret_size_per_runonslice)) for i in [0,1,2]]

        ret_3d_cube = np.zeros(tuple(DATA.shape[:3]) , dtype="float32") # shape = (312, 344, 312)
        for i in range(n_runs_p_dim[0]):
            print("COMPLETION =", 100. * i / n_runs_p_dim[0],"%")
            for j in range(n_runs_p_dim[1]):
                for k in range(n_runs_p_dim[2]): 
                    offset = (ret_size_per_runonslice * i, ret_size_per_runonslice * (j), ret_size_per_runonslice * k)
                    daa = DATA[offset[0] : input_s + offset[0], offset[1] :  input_s + offset[1], offset[2] : input_s + offset[2], :]
                    ret = self.unet.predict(daa) 

                    ret_3d_cube[offset[0] : ret_size_per_runonslice + offset[0], offset[1] : ret_size_per_runonslice + offset[1], offset[2] : ret_size_per_runonslice + offset[2]] = ret[0]


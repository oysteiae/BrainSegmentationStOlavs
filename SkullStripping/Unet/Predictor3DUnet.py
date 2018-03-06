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
        self.unet = build_3DUnet(self.input_size)
        self.unet.load_weights(save_name + ".h5")

    def predict(self):
        for i in range(0, len(self.data)):
            print("Predicting file:", self.d[i])
            pred = self.predict_block(self.data[i])
            helper.save_prediction("unet", pred, "unet", False)

    # TODO: This needs implemented properly.
    def predict_block(self, DATA):
        if self.input_size[0] == DATA.shape[0]:
            daa = np.expand_dims(DATA, axis=0)
            return self.unet.predict(daa)
        else:
            target_labels_per_dim = (DATA).shape[:3]
            ret_size_per_runonslice = self.input_size[0]
            n_runs_p_dim = [int(round(target_labels_per_dim[i] / ret_size_per_runonslice)) for i in [0,1,2]]
    
            DATA = self.greyvalue_data_padding(DATA, 0, 0)

            #ret_3d_cube = np.zeros(tuple(DATA.shape[:3] * np.array([2, 2, 2])) , dtype="float32")
            ret_3d_cube = np.zeros(tuple(DATA.shape[:3]) , dtype="float32")
            print(ret_3d_cube.shape)
            for i in range(n_runs_p_dim[0]):
                print("COMPLETION =", 100. * i / n_runs_p_dim[0],"%")
                for j in range(n_runs_p_dim[1]):
                    for k in range(n_runs_p_dim[2]): 
                        offset = (ret_size_per_runonslice * i, ret_size_per_runonslice * j, ret_size_per_runonslice * k)
                    
                        daa = DATA[offset[0] : ret_size_per_runonslice + offset[0], offset[1] :  ret_size_per_runonslice + offset[1], offset[2] : ret_size_per_runonslice + offset[2], :]
                        #daa = np.expand_dims(daa, axis=0)
                        daa = daa.reshape((1,) + daa.shape)

                        ret = self.unet.predict(daa)
                        #ret = self.run_on_slice(daa)
                        #helper.save_prediction(str(i) + "" + str(j) + "" + str(k) + "daa", np.squeeze(daa), "")
                        #helper.save_prediction(str(i) + "" + str(j) + "" + str(k) + "ret", np.squeeze(ret[0, :, :, :, 0]), "")
                        print(ret.shape)
                        ret_3d_cube[offset[0] : ret_size_per_runonslice + offset[0], offset[1] : ret_size_per_runonslice + offset[1], offset[2] : ret_size_per_runonslice + offset[2]] = np.squeeze(ret[0, :, :, :, 0])
        
        return ret_3d_cube

    def run_on_slice(self, DATA):
        # TODO: Maybe define these for the class
        n_classes = 2
        CNET_stride = [2, 2, 2]
        pred_size = np.array([40, 40, 40])
        DATA = DATA.reshape((1,) + DATA.shape) 
     
        # TODO reimplement the stride stuff
        pred = np.zeros((n_classes,) + tuple(CNET_stride * pred_size),dtype=np.float32) # shape = (2, 32, 32, 32)
        for x in range(CNET_stride[0]):
            for y in range(CNET_stride[1]):
                for z in range(CNET_stride[2]):
                    rr = self.unet.predict(DATA)
                    print(rr.shape)
                    print(pred.shape)
                    pred[0, x::CNET_stride[0], y::CNET_stride[1], z::CNET_stride[2]] = rr[:,:,:,:, 0].reshape((pred_size[0], pred_size[1], pred_size[2])) # shape = (16, 16, 16)
    
        return pred

    def greyvalue_data_padding(self, DATA, offset_l, offset_r):
        avg_value = 1. / 6. * (np.mean(DATA[0]) + np.mean(DATA[:,0]) + np.mean(DATA[:,:,0]) + np.mean(DATA[-1]) + np.mean(DATA[:,-1]) + np.mean(DATA[:,:,-1]))
        sp = DATA.shape
    
        dat = avg_value * np.ones((sp[0] + offset_l + offset_r, sp[1] + offset_l + offset_r, sp[2] + offset_l + offset_r) + tuple(sp[3:]), dtype="float32")
        dat[offset_l : offset_l + sp[0], offset_l : offset_l + sp[1], offset_l : offset_l + sp[2]] = DATA.copy()
    
        return dat

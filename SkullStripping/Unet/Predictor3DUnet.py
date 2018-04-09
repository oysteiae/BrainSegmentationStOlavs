import nibabel as nib
import numpy as np
from Unet.Build3DUnet import build_3DUnet
import helper
from sklearn.feature_extraction import image
import tensorflow as tf

class Predictor3DUnet:
    """description of class"""
    def __init__(self, save_name, file_location, input_size, gpus):
        self.d = helper.load_files(file_location)
        self.save_name = save_name
        self.data = helper.process_data(self.d)
        self.input_size = input_size
        self.unet = build_3DUnet(self.input_size, gpus)
        self.unet.load_weights(save_name + ".h5")

    def predict_data(self):
        for i in range(0, len(self.data)):
            print("Predicting file:", self.d[i])
            #pred = self.predict_block(self.data[i])
            #pred = self.patch_wise_prediction(self.unet,
            #np.expand_dims(np.expand_dims(np.squeeze(self.data[i]), axis=0),
            #axis=4), batch_size=8)
            #pred = self.patch_wise_prediction(self.unet,
            #np.expand_dims(np.squeeze(self.data[i]), axis=0), batch_size=8)
            #pred = self.patch_wise_prediction(self.unet,
            #np.squeeze(self.data[i]), batch_size=8)
            #pred = self.patch_wise_prediction(self.unet, self.data[i],
            #batch_size=8)
            pred = predict_from_patches(self.unet, self.data[i], self.input_size)
            print(pred.shape)
            helper.save_prediction("unet", pred, "unet", False)
                
def predict_from_patches(model, data, input_size):
    predictions = []
    offs = []
    date_shape = data.shape[:3]
    pred_data = np.zeros((date_shape[0], date_shape[1], date_shape[2], 1), dtype="float32")
    n = 100
    overlap = 30
    offs = compute_offs(date_shape, input_size[:3], overlap)
    print(data.shape)
    for r in range(0, len(offs)):
        print("Completed", float(r) / len(offs) * 100)
        #dat = get_segment(data, input_size, data.shape[:3], offs[r])
        patches = tf.extract_image_patches(images=data[0, :, :, :], 
                                           ksizes=input_size[:3], 
                                           strides=[30, 30, 30], 
                                           rates= [1, 1, 1],
                                           padding="VALID")
        print(len(patches))

        #nin = nib.Nifti1Image(dat[0], None, None)
        #nin.to_filename(helper.get_parent_directory() + "/predicted/" + str(r)
        #+ "dat.nii.gz")
        
        #nin = nib.Nifti1Image(pred[0, :, :, :, 1], None, None)
        #nin.to_filename(helper.get_parent_directory() + "/predicted/" + str(r)
        #+ "pred.nii.gz")
        predictions.append(pred)
        #reconstruct_3D_image_from_patch(pred_data, pred, offs[r], input_size,
        #overlap)
    
    reconstruct_whole_3D_image(predictions, offs, pred_data, input_size)
    return pred_data

def compute_offs(data_shape, input_shape, overlap):
    offs = []
    
    step1 = input_shape[0] - overlap
    step2 = input_shape[1] - overlap
    step3 = input_shape[2] - overlap

    for i in range(0, data_shape[0] - input_shape[0], step1):
        for j in range(0, data_shape[1] - input_shape[1], step2):
            for k in range(0, data_shape[2] - input_shape[2], step3):
                offs.append([i, j, k])

    return offs

def get_segment(data, input_shape, data_shape, off):
    dat = np.zeros((1, input_shape[0], input_shape[1], input_shape[2], 1), dtype="float32")
    dat[0,...] = data[off[0] : off[0] + input_shape[0], off[1] : off[1] + input_shape[1], off[2] : off[2] + input_shape[2], :]
    return dat

def predict_batch(model, batch):
    return model.predict(batch)

def reconstruct_3D_image_from_patch(data, predictions, offs, input_shape, overlap):
    predictions[:, -overlap : , -overlap : , -overlap : , 1] = predictions[:, -overlap : , -overlap : , -overlap : , 1] / 2
    predictions[:, : overlap, : overlap, : overlap , 1] = predictions[:, : overlap, : overlap, : overlap , 1] / 2
    
    data[offs[0] : offs[0] + input_shape[0], offs[1] : offs[1] + input_shape[1], offs[2] : offs[2] + input_shape[2], :] = data[offs[0] : offs[0] + input_shape[0], offs[1] : offs[1] + input_shape[1], offs[2] : offs[2] + input_shape[2], :] + np.expand_dims(predictions[0, :, :, :, 1], axis=4)

def reconstruct_whole_3D_image(predictions, offs, data, input_shape):
    for i in range(0, len(predictions)):
        data[offs[i][0] : offs[i][0] + input_shape[0], offs[i][1] : offs[i][1] + input_shape[1], offs[i][2] : offs[i][2] + input_shape[2]] = np.expand_dims(predictions[i][0, :, :, :, 1], axis=4)

def reconstruct_3D_image_from_patches(model, predictions, offs, date_shape, input_shape):
    data = np.zeros((date_shape[0], date_shape[1], date_shape[2]), dtype="float32")
    print(len(offs))
    for i in range(0, len(predictions)):
        average = (data[offs[i][0] : offs[i][0] + input_shape[0], offs[i][1] : offs[i][1] + input_shape[1], offs[i][2] : offs[i][2] + input_shape[2]] + predictions[i][:, :, :, 1]) / 2
        data[offs[i][0] : offs[i][0] + input_shape[0], offs[i][1] : offs[i][1] + input_shape[1], offs[i][2] : offs[i][2] + input_shape[2]] = average
    
    return data
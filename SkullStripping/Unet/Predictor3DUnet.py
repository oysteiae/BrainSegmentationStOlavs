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

    def predict_data(self):
        for i in range(0, len(self.data)):
            print("Predicting file:", self.d[i])
            #pred = self.predict_block(self.data[i])
            #pred = self.patch_wise_prediction(self.unet, np.expand_dims(np.expand_dims(np.squeeze(self.data[i]), axis=0), axis=4), batch_size=8)
            #pred = self.patch_wise_prediction(self.unet, np.expand_dims(np.squeeze(self.data[i]), axis=0), batch_size=8)
            #pred = self.patch_wise_prediction(self.unet, np.squeeze(self.data[i]), batch_size=8)
            #pred = self.patch_wise_prediction(self.unet, self.data[i], batch_size=8)
            pred = predict_from_patches(self.unet, self.data[i], self.input_size)
            helper.save_prediction("unet", pred, "unet", False)
                
def predict_from_patches(model, data, input_size, batch_size=8):
    predictions = []
    offs = []
    for i in range(0, 10000):
        print(data.shape)
        batch, batch_offs = get_batch(data, batch_size, input_size, data.shape[:3])
        pred = predict_batch(model, batch)
        for j in range(0, batch_size):
            predictions.append(pred[j])
            offs.append(batch_offs[j])

    return reconstruct_3D_image_from_patches(model, predictions, offs, data.shape[:3], input_size)

# Returns random batch
def get_batch(data, batch_size, input_shape, data_shape):
    batch = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2], 1))
    offs = []
    print(data_shape)
    for i in range(0, batch_size):
        dat = np.zeros((1, input_shape[0], input_shape[1], input_shape[2], 1), dtype="float32")
        off = [np.random.randint(0, data_shape[x] - input_shape[x]) for x in range(0, 3)]
        offs.append(off)
        dat[0,...] = data[off[0] : off[0] + input_shape[0], off[1] : off[1] + input_shape[1], off[2] : off[2] + input_shape[2], :]
        batch[i] = dat

    return batch, offs

def predict_batch(model, batch):
    return model.predict(batch)

def reconstruct_3D_image_from_patches(model, predictions, offs, date_shape, input_shape):
    data = np.zeros((date_shape[0], date_shape[1], date_shape[2]), dtype="float32")
    print(len(offs))
    for i in range(0, len(predictions)):
        average = (data[offs[i][0] : offs[i][0] + input_shape[0], offs[i][1] : offs[i][1] + input_shape[1], offs[i][2] : offs[i][2] + input_shape[2]] + predictions[i][:, :, :, 1])/2
        data[offs[i][0] : offs[i][0] + input_shape[0], offs[i][1] : offs[i][1] + input_shape[1], offs[i][2] : offs[i][2] + input_shape[2]] = average
    
    return data
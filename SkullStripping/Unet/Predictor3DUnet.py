import nibabel as nib
import numpy as np
from Unet.Build3DUnet import build_3DUnet
import helper
from sklearn.feature_extraction import image

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
            pred = test_predict_from_patches(self.unet, self.data[i], self.input_size[:3])
            #pred = predict_from_patches(self.unet, self.data[i], self.input_size)
            print(pred.shape)
            helper.save_prediction("unet", pred, "unet", False)

def test_predict_from_patches(model, data, input_size, batch_size=1, overlap=2):
    #patch_shape = tuple([int(dim) for dim in model.input.shape[-4:]])
    data = np.squeeze(data)
    patch_shape = input_size
    predictions = list()
    indices = compute_patch_indices(data.shape, patch_size=patch_shape, overlap=overlap)
    batch = list()
    i = 0

    output_shape = [int(model.output.shape[1])] + list(data.shape[-3:])
    print(output_shape)

    while i < len(indices):
        print(indices[i])
        while len(batch) < batch_size:
            patch = get_patch_from_3d_data(data, patch_shape=patch_shape, patch_index=indices[i])
            batch.append(patch)
            i += 1
        prediction = predict_batch(model, np.asarray(batch))
        batch = list()
        for predicted_patch in prediction:
            predictions.append(predicted_patch)
    output_shape = [int(model.output.shape[1])] + list(data.shape[-3:])
    print(output_shape)
    return reconstruct_from_patches(predictions, patch_indices=indices, data_shape=output_shape)

def predict_batch(model, batch):
    return model.predict(np.expand_dims(batch, axis=4))

def compute_patch_indices(image_shape, patch_size, overlap, start=None):
    overlap = np.asarray([overlap] * len(image_shape))
    n_patches = np.ceil(image_shape / (patch_size - overlap))
    overflow = (patch_size - overlap) * n_patches - image_shape + overlap
    start = -np.ceil(overflow/2)
    
    stop = image_shape + start
    step = patch_size - overlap
    
    return get_set_of_patch_indices(start, stop, step)

def get_set_of_patch_indices(start, stop, step):
    return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                               start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)

def get_patch_from_3d_data(data, patch_shape, patch_index):
    patch_index = np.asarray(patch_index, dtype=np.int16)
    patch_shape = np.asarray(patch_shape)
    image_shape = data.shape[-3:]
    if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
        data, patch_index = fix_out_of_bound_patch_attempt(data, patch_shape, patch_index)
    return data[..., patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
                patch_index[2]:patch_index[2]+patch_shape[2]]
  
def fix_out_of_bound_patch_attempt(data, patch_shape, patch_index, ndim=3):
    image_shape = data.shape[-ndim:]
    pad_before = np.abs((patch_index < 0) * patch_index)
    pad_after = np.abs(((patch_index + patch_shape) > image_shape) * ((patch_index + patch_shape) - image_shape))
    pad_args = np.stack([pad_before, pad_after], axis=1)
    if pad_args.shape[0] < len(data.shape):
        pad_args = [[0, 0]] * (len(data.shape) - pad_args.shape[0]) + pad_args.tolist()
    data = np.pad(data, pad_args, mode="edge")
    patch_index += pad_before
    return data, patch_index

def reconstruct_from_patches(patches, patch_indices, data_shape, default_value=0):
    data = np.ones(data_shape) * default_value
    image_shape = data_shape[-3:]
    count = np.zeros(data_shape, dtype=np.int)
    i = 0
    num_iter = len(patches)
    for patch, index in zip(patches, patch_indices):
        print("Completed " + str(float(i)/num_iter) + "%")
        image_patch_shape = patch.shape[-3:]
        if np.any(index < 0):
            fix_patch = np.asarray((index < 0) * np.abs(index), dtype=np.int)
            patch = patch[..., fix_patch[0]:, fix_patch[1]:, fix_patch[2]:]
            index[index < 0] = 0
        if np.any((index + image_patch_shape) >= image_shape):
            fix_patch = np.asarray(image_patch_shape - (((index + image_patch_shape) >= image_shape)
                                                        * ((index + image_patch_shape) - image_shape)), dtype=np.int)
            patch = patch[..., :fix_patch[0], :fix_patch[1], :fix_patch[2]]
        patch_index = np.zeros(data_shape, dtype=np.bool)
        patch_index[...,
                    index[0]:index[0]+patch.shape[-3],
                    index[1]:index[1]+patch.shape[-2],
                    index[2]:index[2]+patch.shape[-1]] = True
        patch_data = np.zeros(data_shape)
        patch_data[patch_index] = patch.flatten()

        new_data_index = np.logical_and(patch_index, np.logical_not(count > 0))
        data[new_data_index] = patch_data[new_data_index]

        averaged_data_index = np.logical_and(patch_index, count > 0)
        if np.any(averaged_data_index):
            data[averaged_data_index] = (data[averaged_data_index] * count[averaged_data_index] + patch_data[averaged_data_index]) / (count[averaged_data_index] + 1)
        count[patch_index] += 1
        i += 1
    
    return data
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
            pred = self.patch_wise_prediction(self.unet, np.expand_dims(np.squeeze(self.data[i]), axis=0), batch_size=8)
            #pred = self.patch_wise_prediction(self.unet, np.squeeze(self.data[i]), batch_size=8)
            #pred = self.patch_wise_prediction(self.unet, self.data[i], batch_size=8)
            print(pred.shape)
            helper.save_prediction("unet", pred, "unet", False)


    def patch_wise_prediction(self, model, data, overlap=0, batch_size=1, permute=False):
        """
        :param batch_size:
        :param model:
        :param data:
        :param overlap:
        :return:
        """
        patch_shape = tuple([int(dim) for dim in model.input.shape[-4:]])
        patch_shape = patch_shape[:3]
        print(patch_shape)
        predictions = list()
        indices = self.compute_patch_indices(data.shape[-3:], patch_size=patch_shape, overlap=overlap)
        batch = list()
        i = 0
        while i < len(indices):
            while len(batch) < batch_size:
                test = indices[i]
                patch = self.get_patch_from_3d_data(data[0], patch_shape=patch_shape, patch_index=indices[i])
                batch.append(patch)
            
            # This is wrongly placed. It should be in the above while loop, but that obuiosly won't work.  
            i += 1
            prediction = self.predict(model, np.asarray(batch), permute=permute)
            batch = list()
            for predicted_patch in prediction:
                predictions.append(predicted_patch)
        output_shape = [int(model.output.shape[1])] + list(data.shape[-3:])
        print("output_shape", output_shape)
        return self.reconstruct_from_patches(predictions, patch_indices=indices, data_shape=output_shape)

    def compute_patch_indices(self, image_shape, patch_size, overlap, start=None):
        print(image_shape)
        print(patch_size)
        if isinstance(overlap, int):
            overlap = np.asarray([overlap] * len(image_shape))
        if start is None:
            n_patches = np.ceil(image_shape / (patch_size - overlap))
            overflow = (patch_size - overlap) * n_patches - image_shape + overlap
            start = -np.ceil(overflow/2)
        elif isinstance(start, int):
            start = np.asarray([start] * len(image_shape))
        stop = image_shape + start
        step = patch_size - overlap
        return self.get_set_of_patch_indices(start, stop, step)


    def get_set_of_patch_indices(self, start, stop, step):
        return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                                   start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)

    def predict(self, model, data, permute=False):
        if permute:
            predictions = list()
            for batch_index in range(data.shape[0]):
                predictions.append(predict_with_permutations(model, data[batch_index]))
            return np.asarray(predictions)
        else:
            data = np.expand_dims(data, axis=4)
            print(data.shape)
            return model.predict(data)

    def get_patch_from_3d_data(self, data, patch_shape, patch_index):
        """
        Returns a patch from a numpy array.
        :param data: numpy array from which to get the patch.
        :param patch_shape: shape/size of the patch.
        :param patch_index: corner index of the patch.
        :return: numpy array take from the data with the patch shape specified.
        """
        patch_index = np.asarray(patch_index, dtype=np.int16)
        patch_shape = np.asarray(patch_shape)
        image_shape = data.shape[-3:]
        if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
            data, patch_index = self.fix_out_of_bound_patch_attempt(data, patch_shape, patch_index)
        return data[..., patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
                    patch_index[2]:patch_index[2]+patch_shape[2]]


    def fix_out_of_bound_patch_attempt(self, data, patch_shape, patch_index, ndim=3):
        """
        Pads the data and alters the patch index so that a patch will be correct.
        :param data:
        :param patch_shape:
        :param patch_index:
        :return: padded data, fixed patch index
        """
        image_shape = data.shape[-ndim:]
        pad_before = np.abs((patch_index < 0) * patch_index)
        
        pad_after = np.abs(((patch_index + patch_shape) > image_shape) * ((patch_index + patch_shape) - image_shape))
        pad_args = np.stack([pad_before, pad_after], axis=1)
        if pad_args.shape[0] < len(data.shape):
            pad_args = [[0, 0]] * (len(data.shape) - pad_args.shape[0]) + pad_args.tolist()
        data = np.pad(data, pad_args, mode="edge")
        patch_index += pad_before
        return data, patch_index

    def reconstruct_from_patches(self, patches, patch_indices, data_shape, default_value=0):
        """
        Reconstructs an array of the original shape from the lists of patches and corresponding patch indices. Overlapping
        patches are averaged.
        :param patches: List of numpy array patches.
        :param patch_indices: List of indices that corresponds to the list of patches.
        :param data_shape: Shape of the array from which the patches were extracted.
        :param default_value: The default value of the resulting data. if the patch coverage is complete, this value will
        be overwritten.
        :return: numpy array containing the data reconstructed by the patches.
        """
        data = np.ones(data_shape) * default_value
        image_shape = data_shape[-3:]
        print("image_shape reconstruct from patches", image_shape)
        print("data", data.shape)
        count = np.zeros(data_shape, dtype=np.int)
        for patch, index in zip(patches, patch_indices):
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
        return data
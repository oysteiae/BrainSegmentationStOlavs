import nibabel as nib
import numpy as np
from Unet.Build3DUnet import build_3DUnet
import helper
import extra
import ntpath

class Predictor3DUnet:
    """description of class"""
    def __init__(self, save_name, input_size, gpus, evaluating_with_slurm=False, loss_function='kld', use_validation=False, part_to_test_on=None):
        self.save_name = save_name
        self.input_size = input_size
        self.model, parallel_model = build_3DUnet(self.input_size, gpus, loss_function)
        self.overlap = input_size[1]/2

        self.use_validation=use_validation
        self.part_to_test_on = part_to_test_on

        helper.load_weights_for_experiment(self.model, save_name, evaluating_with_slurm)

    def predict(self, file_location):
        d = np.asarray(helper.load_files(file_location))
        
        if(self.use_validation and part_to_test_on is not None):
            data = helper.process_data(d[helper.load_indices(self.save_name, part_to_test_on, False)], True)
        else:
            data = helper.process_data(d, True)

        data = helper.process_data(d)
        for i in range(0, len(data)):
            print("Predicting file:", d[i])
            pred = self.predict_data(self.model, data[i], self.input_size[:3])

            helper.save_prediction(ntpath.basename(d[i]).split('.')[0], pred, self.save_name + "_pred_", False, d[i])

    def predict_data(self, model, data, input_size):
        data = np.squeeze(data)
        patch_shape = input_size
        indices = self.compute_patch_indices(data.shape, patch_size=patch_shape, overlap=self.overlap)
        predictions = []
        num_iter = len(indices)
    
        for i in range(num_iter):
            print("Completed " + "%.2f" % ((float(i) / num_iter) * 100) + "% of predicting patches")
            patch = self.get_patch(data, patch_shape=patch_shape, patch_index=indices[i])
            prediction = self.predict_patch(model, patch)

            # The model outputs one channel and two classes.  We only need the
            # data, therefore only one class is chosen
            predictions.append(prediction[0, :, :, :, 0])
    
        return self.reconstruct_data_from_predicted_patches(predictions, patch_indices=indices, data_shape=data.shape)

    def predict_patch(self, model, patch):
        # The data has to be expanded so that it fits the keras model.
        return model.predict(np.expand_dims(np.expand_dims(patch, axis=4), axis=0))

    def compute_patch_indices(self, data_shape, patch_size, overlap):
        # These still need to be rewritten
        overlap = np.asarray([overlap] * len(data_shape))
        n_patches_in_each_dim = np.ceil(data_shape / (patch_size - overlap))
        overflow = (patch_size - overlap) * n_patches_in_each_dim - data_shape + overlap
        start = -np.ceil(overflow / 2)
        step = patch_size - overlap
    
        patch_indices = []
        for i in range(0, abs(int(n_patches_in_each_dim[0]))):
            for j in range(0, abs(int(n_patches_in_each_dim[1]))):
                for k in range(0, abs(int(n_patches_in_each_dim[2]))):
                    patch_indices.append([start[0] + step[0] * i, start[1] + step[1] * j, start[2] + step[2] * k])

        return np.asarray(patch_indices, dtype=np.int)

    def get_patch(self, data, patch_shape, patch_index):
        patch_index = np.asarray(patch_index, dtype=np.int16) # This copies it so that the original is untouched
        data_shape = data.shape
    
        # This handles if the patch indices are less than 0 or larger than the data
        # shape
        if(np.any(patch_index < 0) or np.any(data_shape < (patch_index + patch_shape))):
            # (nbefore, nafter) in each dimension
            #npad = ((abs((patch_index[0] < 0) * patch_index[0]), ((patch_index[0] + patch_shape[0] - data_shape[0]) > 0) * (patch_index[0] + patch_shape[0] - data_shape[0])), 
            #        (abs((patch_index[1] < 0) * patch_index[1]), ((patch_index[1] + patch_shape[1] - data_shape[1]) > 0) * (patch_index[1] + patch_shape[1] - data_shape[1])), 
            #        (abs((patch_index[2] < 0) * patch_index[2]), ((patch_index[2] + patch_shape[2] - data_shape[2]) > 0) * (patch_index[2] + patch_shape[2] - data_shape[2])))
            # Pads with the edge values of array.
            before = max(abs((patch_index[0] < 0) * patch_index[0]), abs((patch_index[1] < 0) * patch_index[1]), abs((patch_index[2] < 0) * patch_index[2]))
            after = max(((patch_index[0] + patch_shape[0] - data_shape[0]) > 0) * (patch_index[0] + patch_shape[0] - data_shape[0]), 
                        ((patch_index[1] + patch_shape[1] - data_shape[1]) > 0) * (patch_index[1] + patch_shape[1] - data_shape[1]),
                        ((patch_index[2] + patch_shape[2] - data_shape[2]) > 0) * (patch_index[2] + patch_shape[2] - data_shape[2]))
            #data = np.pad(data, npad, mode='edge')
            data = extra.greyvalue_data_padding(data, before, after)
            # The patch index has to be updated so that the negative indexes are
            # "fixed" so that it handles the padded data.
            patch_index += np.asarray((abs((patch_index[0] < 0) * patch_index[0]), abs((patch_index[1] < 0) * patch_index[1]), abs((patch_index[2] < 0) * patch_index[2])))

        return data[patch_index[0] : patch_index[0] + patch_shape[0], 
                    patch_index[1] : patch_index[1] + patch_shape[1],
                    patch_index[2] : patch_index[2] + patch_shape[2]]
  
    def reconstruct_data_from_predicted_patches(self, patches, patch_indices, data_shape):
        data = np.zeros(data_shape)
        count = np.zeros(data_shape, dtype=np.int)
    
        i = 0
        num_iter = len(patches)
        for patch, index in zip(patches, patch_indices):
            print("Completed " + "%.2f" % ((float(i) / num_iter) * 100) + "% of rebuilding image from predicted patches")
        
            orig_patch_shape = np.copy(patch.shape)
            #Fix index and patch out of bounds
            if(np.any(index < 0)):
                patch = patch[((index[0] < 0) * abs(index[0])) : , 
                              ((index[1] < 0) * abs(index[1])) : , 
                              ((index[2] < 0) * abs(index[2])) : ]
                index[index < 0] = 0
            if(np.any((index + orig_patch_shape) >= data_shape)):
                patch = patch[: orig_patch_shape[0] - (((index[0] + orig_patch_shape[0] - data_shape[0]) >= 0) * (index[0] + orig_patch_shape[0] - data_shape[0])), 
                              : orig_patch_shape[1] - (((index[1] + orig_patch_shape[1] - data_shape[1]) >= 0) * (index[1] + orig_patch_shape[1] - data_shape[1])), 
                              : orig_patch_shape[2] - (((index[2] + orig_patch_shape[2] - data_shape[2]) >= 0) * (index[2] + orig_patch_shape[2] - data_shape[2]))]
        
            patch_index = np.zeros(data_shape, dtype=np.bool)
            patch_index[index[0] : index[0] + patch.shape[0],
                        index[1] : index[1] + patch.shape[1],
                        index[2] : index[2] + patch.shape[2]] = True
        
            patch_data = np.zeros(data_shape)
            patch_data[patch_index] = patch.flatten()

            indeces_where_data_not_already_added = np.logical_and(patch_index, np.logical_not(count > 0))
            data[indeces_where_data_not_already_added] = patch_data[indeces_where_data_not_already_added]

            averaged_data_index = np.logical_and(patch_index, count > 0)
            if np.any(averaged_data_index):
                data[averaged_data_index] = (data[averaged_data_index] * count[averaged_data_index] + patch_data[averaged_data_index]) / (count[averaged_data_index] + 1)
        
            # count where values have been placed
            count[patch_index] += 1
            i += 1

        return data
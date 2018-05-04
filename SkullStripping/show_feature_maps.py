from keras import models
import helper
from CNN import Build3DCNN
import extra
import matplotlib.pyplot as plt
import numpy as np


def main():
    model, paralell_model = Build3DCNN.build_3DCNN((176, 176, 176, 1), 1)
    model.load_weights("CNNAll.h5")
    
    layer_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    
    #d = helper.load_files(["D:\\MRISCANS\\da\\"])
    d = helper.load_files(["C:\\Users\\oyste\\Documents\\MRIScans\\OASIS\\da"])
    data = helper.process_data(d, True)
    
    DATA = data[0]
    print(DATA.shape)
    n_classes = 2
    input_s = 84
    target_labels_per_dim = DATA.shape[:3]
    
    ret_size_per_runonslice = 32
    n_runs_p_dim = [int(round(target_labels_per_dim[i] / ret_size_per_runonslice)) for i in [0,1,2]]
    
    # TODO: Set or calculate these numbers yourself.
    offset_l = 26
    offset_r = 110

    #DATA = extra.greyvalue_data_padding(DATA, offset_l, offset_r)

    ret_3d_cube = np.zeros(tuple(DATA.shape[:3]) , dtype="float32") # shape = (312, 344, 312)
    offset = [40, 40, 40]
    daa = DATA[offset[0] : input_s + offset[0], offset[1] :  input_s + offset[1], offset[2] : input_s + offset[2], :]
    activations = activation_model.predict(np.expand_dims(DATA[:, : 176, :], axis=0))

    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)
        images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, 10, :, :, col * images_per_row + row]
                #channel_image -= channel_image.mean()
                #channel_image /= channel_image.std()
                #channel_image *= 64
                #channel_image += 128
                #channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
    
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='gray')
        plt.show()
    #print(len(activations))
    #first_layer_activation = activations[7]
    #print(first_layer_activation.shape)
    #plt.matshow(first_layer_activation[0, 5, :, : , 0], cmap='gray')
    #plt.show()

main()
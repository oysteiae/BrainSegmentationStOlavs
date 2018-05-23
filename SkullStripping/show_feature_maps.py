from keras import models
import helper
from CNN import Build3DCNN
from Unet import Build3DUnet
import extra
import matplotlib.pyplot as plt
import numpy as np
import math

def main():
    cube_size = 176

    #model, paralell_model = Build3DCNN.build_3DCNN((176, 176, 176, 1), 1, 'kld')
    model, paralell_model = Build3DUnet.build_3DUnet((cube_size, cube_size, cube_size, 1), 1, 'kld')
    model.load_weights("C:\\Users\\oyste\\Documents\\Visual Studio 2015\\Projects\\SkullStripping\\BrainSegmentationStOlavs\\Experiments\\UnetAll\\UnetAll.h5")
    
    layer_outputs = [layer.output for layer in model.layers[1:8]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    
    d = helper.load_files(["D:\\MRISCANS\\predict\\"])
    #d = helper.load_files(["C:\\Users\\oyste\\Documents\\MRIScans\\OASIS\\da"])
    DATA = helper.load_file_as_nib(d[0])
    
    activations = activation_model.predict(np.expand_dims(DATA[: cube_size, : cube_size, : cube_size], axis=0))

    layer_names = []
    for layer in model.layers[1:8]:
        layer_names.append(layer.name)
    
    images_per_row = 8
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = math.ceil(float(n_features) / float(images_per_row))
        print(n_cols)
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                if(col * images_per_row + row >= n_features):
                    break

                channel_image = layer_activation[0, 10, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
    
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.savefig("D:\\Master\\Graphs\\featuremapsUnet\\" + layer_name + "Unet" + ".png")
        plt.show()
    #print(len(activations))
    #first_layer_activation = activations[7]
    #print(first_layer_activation.shape)
    #plt.matshow(first_layer_activation[0, 5, :, : , 0], cmap='gray')
    #plt.show()

main()
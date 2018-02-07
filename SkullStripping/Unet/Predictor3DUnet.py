import nibabel as nib
from Build3DUnet import build_3DUnet
import helper

class Predictor3DUnet:
    """description of class"""
    def __init__(self, save_name, file_location, input_size=(144, 144, 144), initial_learning_rate=0.001):
        self.d = helper.load_files(file_location)
        self.save_same = save_name
        self.data = helper.process_data(self.d)
        
        self.unet = build_3DUnet(input_shape, use_upsampling=False)
        self.unet.load_weights(save_name + ".h5")
        
    def predict(self):
        for i in range(0, len(data)):
            prediction = self.unet.predict(self.data[i])

            helper.save_prediction(self.save_name, prediction, d[i])
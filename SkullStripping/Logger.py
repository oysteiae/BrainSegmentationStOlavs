import keras
import time

#TODO also log validation set loss.
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []
        self.timestamp = []
        self.t0 = time.time()

    #def on_batch_end(self, batch, logs={}):
        #self.losses.append(logs.get('loss'))
        #self.accuracies.append(logs.get('acc'))
    
    def on_epoch_end(self, epoch, logs = None):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))
        self.timestamp.append(time.time() - self.t0)
        
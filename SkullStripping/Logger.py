import keras
import time

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []
        self.timestamp = []
        self.t0 = time.time()
        print("Halllllllll")
        print("tuddelidei")

    #def on_batch_end(self, batch, logs={}):
        #self.losses.append(logs.get('loss'))
        #self.accuracies.append(logs.get('acc'))
    
    def on_epoch_end(self, epoch, logs = None):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))
        print(time.time() - self.t0)
        self.timestamp.append(time.time() - self.t0)
        
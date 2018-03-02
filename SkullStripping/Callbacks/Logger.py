import keras
import time

#TODO: What happens when no validation data is available?
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

        self.accuracies = []
        self.val_accuracies = []
        
        self.timestamp = []
        self.t0 = time.time()
        self.accumulated_acc = 0

    #def on_batch_end(self, batch, logs={}):
        #self.losses.append(logs.get('loss'))
        #self.accuracies.append(logs.get('acc'))
    
    def on_epoch_end(self, epoch, logs = None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
        self.accuracies.append(logs.get('acc'))
        self.val_accuracies.append(logs.get('val_acc'))
        
        self.timestamp.append(time.time() - self.t0)
        self.accumulated_acc += self.accuracies.append(logs.get('acc'))

        if(epoch % 100 == 0):
            print("Average accuracy from epoch", epoch, "to epoch", epoch, "is:", self.accumulated_acc/100.0)
            self.accumulated_acc = 0
        
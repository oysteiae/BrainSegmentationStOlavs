import keras
import keras.backend as K

# TODO change lame name
class MonitorStopping(keras.callbacks.Callback):
    """description of class"""
    def __init__(self, model):
        self.model = model
        self.best_training_loss_epoch = 0
        self.best_training_loss = 1000
        self.last_learning_rate_decrease_step = 0
        self.reduction_steps_so_far = 0

    def on_epoch_end(self, epoch, logs={}):
        if(self.best_training_loss > logs.get('loss')):
            self.best_training_loss_epoch = epoch
            self.best_training_loss = logs.get('loss')
        else:
            if(epoch - self.best_training_loss_epoch > 5000 and epoch - self.last_learning_rate_decrease_step >= 4000):
                self.last_learning_rate_decrease_step = epoch
                lr = K.get_value(self.model.optimizer.lr)
                K.set_value(self.model.optimizer.lr, lr*0.5)
                self.reduction_steps_so_far += 1

                if(self.reduction_steps_so_far == 10):
                    self.model.stop_training = True

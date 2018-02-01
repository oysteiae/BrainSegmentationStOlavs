# TODO rename class
from keras import backend as K

#dice_coefficient = (2 * TP) / (2 * TP + FP + FN)
# Taken from https://github.com/ellisdg/3DUnetCNN/blob/36a321e1ca36fd22845067569b0ae471ababb096/unet3d/metrics.py
def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)
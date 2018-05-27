# TODO rename class
from keras import backend as K
import numpy as np

#dice_coefficient = (2 * TP) / (2 * TP + FP + FN)
# Taken from
# https://github.com/ellisdg/3DUnetCNN/blob/36a321e1ca36fd22845067569b0ae471ababb096/unet3d/metrics.py
def dice_coefficient(y_true, y_pred, smooth=1., threshold=0.5):
    #y_true_f = K.flatten(y_true)
    #y_pred_f = K.flatten(y_pred)
    #intersection = K.sum(y_true_f * y_pred_f)
    #return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    # If axis is negative it counts from the last to the first axis.

    # Final test if thresholding will work.
    #y_pred = K.cast(y_pred > threshold, dtype="float32")
    #y_true = K.cast(y_true > threshold, dtype="float32")

    #y_pred = y_pred > threshold
    #y_true = y_true > threshold

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

# Taken from: https://github.com/GUR9000/Deep_MRI_brain_extraction
# Pads the data with the averatge data
def greyvalue_data_padding(DATA, offset_l_and_t, offset_r_and_b):
    avg_value = 1. / 6. * (np.mean(DATA[0]) + np.mean(DATA[:,0]) + np.mean(DATA[:,:,0]) + np.mean(DATA[-1]) + np.mean(DATA[:,-1]) + np.mean(DATA[:,:,-1]))
    sp = DATA.shape
    
    dat = avg_value * np.ones((sp[0] + offset_l_and_t + offset_r_and_b, sp[1] + offset_l_and_t + offset_r_and_b, sp[2] + offset_l_and_t + offset_r_and_b) + tuple(sp[3:]), dtype="float32")
    dat[offset_l_and_t : offset_l_and_t + sp[0], offset_l_and_t : offset_l_and_t + sp[1], offset_l_and_t : offset_l_and_t + sp[2]] = DATA.copy()
    
    return dat
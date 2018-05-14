import numpy as np

# The offset on the input for the central predicted voxel. 
# 53 is the receptive field for the model, if the model is changed later this should
# be reimplemented to account for a change in the receptive field.
def compute_label_offset():
    off = (int)(53 - 1) / 2
    return np.array((off, off, off), dtype='int32')
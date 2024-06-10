import numpy as np
import scipy.io
import os

# this function gets list of paths to .mat files and calculates max length of read (number of samples)
def get_max_len(files_to_process):
    max_value = float('-inf')
    for _, element_path in enumerate(files_to_process):
        file = scipy.io.loadmat(element_path)
        if len(file["Rcr"]) > max_value:
            max_value = len(file["Rcr"])
    return max_value

# this function gets loaded .mat file and needed size of read, pads arrays - return shape (1801, 12)
def expand_element(file, measured_quantity, needed_size):
    meas_array = file[measured_quantity]
    no_elements_to_pad = needed_size - len(meas_array[:,0])
    rows, cols = meas_array.shape
    lisf_for_padded_cols = []
    for i in range(cols):
        lisf_for_padded_cols.append(np.pad(meas_array[:,i], (0,no_elements_to_pad), 'constant', constant_values = 0))
    return np.array(lisf_for_padded_cols)

# this function takes path to a file and returns category eg. 'bike' or 'car'
def category(filepath):
    dirname = os.path.dirname(filepath)
    category = os.path.basename(dirname)
    return category

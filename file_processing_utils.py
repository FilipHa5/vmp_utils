import numpy as np
import scipy.io
import os
import tqdm


# this function takes path to a file and returns category eg. 'bike' or 'car'
def category(filepath):
    filename = os.path.basename(filepath)
    dirname = os.path.dirname(filepath)
    category = os.path.basename(dirname)
    return filename, category


def get_file(path):
    file = scipy.io.loadmat(path)
    filename, vehicle_category = category(path)
    return {
        "filename": filename,
        "content": file,
        "len": len(file["Rcr"]),
        "vehicle_category": vehicle_category,
    }


# this function gets list of paths to .mat files and calculates max length of read (number of samples)
def get_max_len(files_to_process):
    files = []
    max_value = float("-inf")
    categories = set()
    for element_path in tqdm.tqdm(files_to_process, desc="Reading files"):
        file = get_file(element_path)
        files.append(file)
        categories.add(file["vehicle_category"])
        if file["len"] > max_value:
            max_value = file["len"]
    return max_value, files, categories


def expand_element(file, measured_quantity, needed_len):
    """
    Loads a specific measurement array from a .mat file, and pads it with zeros 
    to match the required number of rows. Returns the result as a NumPy array 
    of shape (needed_len, qty_of_channels).
    
    Parameters:
        file (dict): Loaded .mat file (typically from scipy.io.loadmat).
        measured_quantity (str): Key in the file corresponding to the desired array.
        needed_len (int): Desired number of rows in the output array.
        
    Returns:
        np.ndarray: Padded array of shape (needed_len, qty_of_channels), dtype float32.
    """
    meas_array = file[measured_quantity]
    no_elements_to_pad = needed_len - len(meas_array[:, 0])
    rows, cols = meas_array.shape
    lisf_for_padded_cols = []
    for i in range(cols):
        lisf_for_padded_cols.append(
            np.pad(
                meas_array[:, i], (0, no_elements_to_pad), "constant", constant_values=0
            )
        )
    return np.array(lisf_for_padded_cols)

# a może
    # return np.pad(
    #     meas_array.T, 
    #     ((0, 0), (0, no_elements_to_pad)), 
    #     mode="constant", 
    #     constant_values=0
    # )

def prepare_expanded_file(file, len):
    expanded_reads_rcr = expand_element(file["content"], "Rcr", len)
    expanded_reads_xcr = expand_element(file["content"], "Xcr", len)
    return file["filename"], expanded_reads_rcr, expanded_reads_xcr

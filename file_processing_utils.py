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


def get_file_len(path):
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
        file = get_file_len(element_path)
        files.append(file)
        categories.add(file["vehicle_category"])
        if file["len"] > max_value:
            max_value = file["len"]
    return max_value, files, categories


# this function gets loaded .mat file and needed size of read, pads arrays - return shape (1801, 12)
def expand_element(file, measured_quantity, needed_size):
    meas_array = file[measured_quantity]
    no_elements_to_pad = needed_size - len(meas_array[:, 0])
    rows, cols = meas_array.shape
    lisf_for_padded_cols = []
    for i in range(cols):
        lisf_for_padded_cols.append(
            np.pad(
                meas_array[:, i], (0, no_elements_to_pad), "constant", constant_values=0
            )
        )
    return np.array(lisf_for_padded_cols)


def prepare_expanded_file(file, len):
    expanded_reads_rcr = expand_element(file["content"], "Rcr", len)
    expanded_reads_xcr = expand_element(file["content"], "Xcr", len)
    return [file["filename"], expanded_reads_rcr, expanded_reads_xcr]

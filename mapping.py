import os
import numpy as np

def get_class_to_filename_map(path_to_files: str) -> dict:
    fileslist = os.listdir(path_to_files)
    class_to_name_map = {index: filename.removesuffix(".npy") for index, filename in enumerate(fileslist)}
    return class_to_name_map

def save_class_to_filename_mapping(path_to_files: str) -> None:
    mapping = get_class_to_filename_map(path_to_files)
    np.save("mapping", mapping)
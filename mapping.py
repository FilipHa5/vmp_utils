import os

def get_class_to_filename_map(path_to_files: str) -> dict:
    fileslist = os.listdir(path_to_files)
    class_names = [filename.rstrip(".npy") for filename in fileslist]
    class_to_name_map = {index: filename.removeprefix(".npy") for index, filename in enumerate(fileslist)}
    return class_to_name_map
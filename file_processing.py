import os
import numpy as np
from glob import glob

from file_processing_utils import (
    get_max_len,
    prepare_expanded_file,
)
from mapping import save_class_to_filename_mapping

MAT_SRC_PATH = "extracted"
NPY_DEST_PATH = "npy"


def process():
    # TODO: check if extracted exists, otherwise break and notify
    files_to_process = [
        y for x in os.walk(MAT_SRC_PATH) for y in glob(os.path.join(x[0], "*_K.mat"))
    ]
    max_len, files, categories = get_max_len(files_to_process)
    print(f"Max length of VMP signal: {max_len} samples")

    categories = list(categories)
    categorized_files = [
        list(filter(lambda x: x["vehicle_category"] == key, files))
        for key in categories
    ]

    for sorted_files in categorized_files:
        signle_cat_npy = []
        for file in sorted_files:
            name_r_x = np.asarray(prepare_expanded_file(file, max_len), dtype=object)
            signle_cat_npy.append(name_r_x)
        if not os.path.exists(NPY_DEST_PATH):
            os.mkdir(NPY_DEST_PATH)
            print("Created destination path", NPY_DEST_PATH)

        npy_file_name = os.path.join(
            NPY_DEST_PATH, sorted_files[0]["vehicle_category"] + ".npy"
        )
        np.save(npy_file_name, signle_cat_npy)
        print(f"Saved file: {npy_file_name}")

    save_class_to_filename_mapping(NPY_DEST_PATH)


if __name__ == "__main__":
    process()
    print(f"Completed! Files saved in {NPY_DEST_PATH}")

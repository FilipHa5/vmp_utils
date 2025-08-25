import os
import numpy as np
from glob import glob
import argparse

from file_processing_utils import (
    get_max_len,
    prepare_expanded_file,
)
from mapping import save_class_to_filename_mapping


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matdir",
        type=str,
        help="Determines directory where .mat files are located. If not specified 'extracted' assumed",
        default="extracted",
    )
    parser.add_argument(
        "--format",
        type=str,
        help="Determines structure and format of resulting aggregated files. npy or npz are allowed. If not specified 'npy' assumed",
        default="npy",
    )
    return parser.parse_args()


def process_to_npy(categorized_files, max_len, DEST_DIRNAME):
    for sorted_files in categorized_files:
        signle_cat_npy = []
        for file in sorted_files:
            name_r_x = np.asarray(prepare_expanded_file(file, max_len), dtype=object)
            signle_cat_npy.append(name_r_x)
        if not os.path.exists(DEST_DIRNAME):
            os.mkdir(DEST_DIRNAME)
            print("Created destination path", DEST_DIRNAME)
        file_name = os.path.join(
            DEST_DIRNAME, sorted_files[0]["vehicle_category"] + "." + DEST_DIRNAME
        )
        np.save(file_name, signle_cat_npy)
        print(f"Saved file: {file_name}")


def process_to_npz(categorized_files, max_len, DEST_DIRNAME):
    for sorted_files in categorized_files:
        filenames, expanded_reads_r, expanded_reads_x = [], [], []
        for file in sorted_files:
            filename, expanded_r, expanded_x = prepare_expanded_file(file, max_len)
            filenames.append(filename)
            expanded_reads_r.append(expanded_r.astype(np.float32))
            expanded_reads_x.append(expanded_x.astype(np.float32))
        if not os.path.exists(DEST_DIRNAME):
            os.mkdir(DEST_DIRNAME)
            print("Created destination path", DEST_DIRNAME)
        reads_r_cat = np.stack(expanded_reads_r)  # shape (N, 12, 3000)
        reads_x_cat = np.stack(expanded_reads_x)  # shape (N, 12, 3000)
        filenames_array = np.array(filenames, dtype="U")  # shape (N,)

        file_name = os.path.join(
            DEST_DIRNAME, sorted_files[0]["vehicle_category"] + "." + DEST_DIRNAME
        )
        np.savez(file_name, filenames=filenames_array, R=reads_r_cat, X=reads_x_cat)

        print(f"Saved file: {file_name}")


def get_categorized_files(categories, files):
    categories = list(categories)
    categorized_files = [
        list(filter(lambda x: x["vehicle_category"] == key, files))
        for key in categories
    ]
    return categorized_files


def main():
    args = parse_args()
    MAT_SRC_PATH = args.matdir
    DEST_DIRNAME = args.format

    # TODO: check if extracted exists, otherwise break and notify
    files_to_process = [
        y for x in os.walk(MAT_SRC_PATH) for y in glob(os.path.join(x[0], "*_K.mat"))
    ]
    max_len, files, categories = get_max_len(files_to_process)
    print(f"Max length of VMP signal: {max_len} samples")

    categorized_files = get_categorized_files(categories, files)

    if DEST_DIRNAME == "npy":
        process_to_npy(categorized_files, max_len, DEST_DIRNAME)
    if DEST_DIRNAME == "npz":
        process_to_npz(categorized_files, max_len, DEST_DIRNAME)

    # TODO: is it needed? Does it work?
    #save_class_to_filename_mapping(DEST_DIRNAME)


if __name__ == "__main__":
    main()

import os
import sys
import numpy as np
from glob import glob
from scipy.io import loadmat

from file_processing_utils import get_max_len, expand_element, category
from mapping import save_class_to_filename_mapping

if __name__ == "__main__":

    mat_src_path = sys.argv[1]
    npy_dest_path = sys.argv[2]

    files_to_process = [y for x in os.walk(mat_src_path) for y in glob(os.path.join(x[0], '*_K.mat'))]

    max_len = get_max_len(files_to_process)
    print(f"Processing {files_to_process} files")
    print(f"Max length of sample: {max_len} samples")

    categories = set()
    for index, filepath in enumerate(files_to_process):
        cat = category(filepath)
        categories.add(cat)

    categories = list(categories)
    print(f"Categories: {categories}")

    # grouped_files_paths is a nested list, where sublists contain paths only to a single category files
    grouped_files_paths = []
    for cat in categories:
        one_cat_paths = [i for i in files_to_process if cat in i]
        grouped_files_paths.append(one_cat_paths)

    for _, list_of_path_to_files_category in grouped_files_paths:
        category_name = category(list_of_path_to_files_category[0])
        print(f"Processing {category_name}")
        signle_cat_npy = []
        for _, filepath in enumerate(list_of_path_to_files_category):
            file = loadmat(filepath)
            expanded_reads_rcr = expand_element(file, "Rcr", max_len)
            expanded_reads_xcr = expand_element(file, "Xcr", max_len)
            name_r_x = [os.path.basename(filepath), expanded_reads_rcr, expanded_reads_xcr]
            name_r_x = np.asarray(name_r_x, dtype=object)
            signle_cat_npy.append(name_r_x)

        category_name = str(category_name)

        if not os.path.exists(npy_dest_path):
            os.mkdir(npy_dest_path)
            print("Created destination path", npy_dest_path)
        else:
            print("Destination path found", npy_dest_path)

        npy_file_name = os.path.join(npy_dest_path, category_name + ".npy")
        np.save(npy_file_name, signle_cat_npy)
        print(f"Saved {category_name}.npy")
    save_class_to_filename_mapping(npy_dest_path)
    print("Completed! Thanks for making this little script happy :)")

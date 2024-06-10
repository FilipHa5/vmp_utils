import os
import sys
import numpy as np
from glob import glob
from scipy.io import loadmat

from file_processing_utils import get_max_len, expand_element, category

if __name__ == "__main__":

    mat_src_path = sys.argv[1]
    npy_dest_path = sys.argv[2]

    files_to_process = [y for x in os.walk(mat_src_path) for y in glob(os.path.join(x[0], '*_K.mat'))]

    max_len = get_max_len(files_to_process)
    print(f"Processing {files_to_process} files")
    print(f"Max length of sample = {max_len} samples")

    categories = set()
    for index, filepath in enumerate(files_to_process):
        cat = category(filepath)
        categories.add(cat)

    categories = list(categories)
    print(f"Categories: {categories}")

    grouped_files_paths = []
    for cat in categories:
        one_cat_paths = [i for i in files_to_process if cat in i]
        grouped_files_paths.append(one_cat_paths)

    for list_of_path_to_files_category in grouped_files_paths:
        category_name = category(list_of_path_to_files_category[0])
        print(f"Processing {category_name}")
        results = []
        for index, filepath in enumerate(list_of_path_to_files_category):
            file = loadmat(filepath)
            expanded_reads_rcr = expand_element(file, "Rcr", max_len)
            expanded_reads_xcr = expand_element(file, "Xcr", max_len)
            pair = [os.path.basename(filepath), expanded_reads_rcr, expanded_reads_xcr]
            pair = np.asarray(pair, dtype=object)
            results.append(pair)

        category_name = str(category_name)

        if not os.path.exists(npy_dest_path):
            os.mkdir(npy_dest_path)
            print("Created destination path", npy_dest_path)
        else:
            print("Destination path found", npy_dest_path)

        npy_file_name = os.path.join(npy_dest_path, category_name + ".npy")
        np.save(npy_file_name, results)
        print(f"Saved {category_name}.npy")
    print("Completed! Thanks for making this little script happy :)")

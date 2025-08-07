import unittest
import numpy as np
from pathlib import Path

class TestNpyDirectoryProperties(unittest.TestCase):
    def setUp(self):
        # Path to directory containing .npy files
        self.npy_dir = Path("npy")
        self.files = [file for file in self.npy_dir.iterdir()]

    def test_existing_npy_directory(self):
        self.assertTrue(self.npy_dir.exists(), f"Directory {self.npy_dir} does not exist. Maybe you are in wrong dir?")

    def test_directory_contains_npy_files(self):
        files_qty = len(self.files)
        self.assertTrue(files_qty > 1, f"Direcory {self.npy_dir} has {files_qty} files.")

class TestNpyMatrixStructure(unittest.TestCase):
    def setUp(self):
        # Path to directory containing .npy files
        self.npy_dir = Path("npy")
        self.expected_shape = (3, )
        self.expected_rx_matrix_shape = (12, 3002)

    def test_npy_file_structure(self):
        npy_files = list(self.npy_dir.glob("*.npy"))
        self.assertGreater(len(npy_files), 0, "No .npy files found.")

        for file_npy in npy_files:
                data = np.load(file_npy, allow_pickle=True)
                for vehicle in data:
                    with self.subTest(file=file_npy.name):
                        try:
                            # Confirm data has at least 3 items
                            self.assertTrue(len(data) > 2, f"{file_npy.name} has less than 3 items")

                            # Confirm data[1] and data[2] are arrays
                            matrix_r = data[1]
                            matrix_x = data[2]

                            self.assertIsInstance(matrix_r, np.ndarray, f"{file_npy.name} - data[1] is not an ndarray")
                            self.assertIsInstance(matrix_x, np.ndarray, f"{file_npy.name} - data[2] is not an ndarray")

                            # Confirm correct shape
                            self.assertEqual(matrix_r.shape, self.expected_shape,
                                            f"{file_npy.name} - data[1] shape is {matrix_r.shape}, expected {self.expected_shape}")
                            self.assertEqual(matrix_x.shape, self.expected_shape,
                                            f"{file_npy.name} - data[2] shape is {matrix_x.shape}, expected {self.expected_shape}")
                        except Exception as e:
                            self.fail(f"Failed to process {file_npy.name}: {e}")

if __name__ == "__main__":
    unittest.main()

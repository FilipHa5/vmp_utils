import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transforms import VMPData
import torch

class TestVMPData(unittest.TestCase):
    def setUp(self):
        self.dataset_path = 'npy'

    def test_loading_real_dir(self):
        dataset = VMPData(self.dataset_path)

        self.assertGreater(len(dataset), 0, "Dataset should not be empty")

        sample = dataset[0]
        data, label, filename = sample

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        self.assertTrue(isinstance(data[0], np.ndarray) or isinstance(data[0], torch.Tensor))
        self.assertIsInstance(label, torch.Tensor)
        self.assertEqual(label.dtype, torch.long)
        self.assertIsInstance(filename, str)

    def test_labels_correspond_to_file_index(self):
        files = sorted([f for f in os.listdir(self.dataset_path) if f.endswith('.npy')])
        dataset = VMPData(self.dataset_path)

        labels_seen = set()
        for i in range(len(dataset)):
            _, label, _ = dataset[i]
            labels_seen.add(label.item())

        expected_labels = set(range(len(files)))
        self.assertEqual(labels_seen, expected_labels)

    def test_with_transform_real_dir(self):
        class DummyTransform:
            def __call__(self, data):
                return [d * 0 for d in data]

        dataset = VMPData(self.dataset_path, transform=DummyTransform())
        data, _, _ = dataset[0]

        for d in data:
            self.assertTrue(np.all(d == 0))

    def test_returns_consistent_length_and_items(self):
        dataset = VMPData(self.dataset_path)
        length = len(dataset)

        for idx in range(min(length, 10)):
            item = dataset[idx]
            self.assertEqual(len(item), 3)
            self.assertIsInstance(item[1], torch.Tensor)


if __name__ == '__main__':
    unittest.main()
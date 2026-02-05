from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch

from transforms import VMPData, VMPDataWideSlim, VMPDataWideSlimCNN


class DataProcessor():
    def __init__(self, path):
        self.path = path
        self.transforms = []

    def add_transforms(self, *transforms):
        self.transforms += transforms

    def compose_transforms(self):
        composed_transforms = transforms.Compose(self.transforms)
        return composed_transforms

    def get_dataset(self, train_ratio=0.7, SEED=None):
        transformed_data = VMPData(self.path, transform = self.compose_transforms())

        # generate train and test indices
        train_indices, test_indices, train_labels, test_labels = train_test_split(
            range(len(transformed_data)),
            transformed_data.labels,
            stratify = transformed_data.labels,
            test_size = 1 - train_ratio,
            random_state = SEED
            )

        print("Train set size:", len(train_indices))
        print("Test set size:",len(test_indices))

        # generate subset based on indices
        dataset_train = Subset(transformed_data, train_indices)
        dataset_test = Subset(transformed_data, test_indices)

        return dataset_train, dataset_test

    def get_dataset_wide_slim(self, train_ratio=0.7, val_ratio=0.1, SEED=None):
        transformed_data = VMPDataWideSlim(self.path, transform=self.compose_transforms())

    def get_dataset_wide_slim_cnn(self, train_ratio=0.7, val_ratio=0.1, SEED=None):
        transformed_data = VMPDataWideSlimCNN(self.path, transform=self.compose_transforms())

        # ---- First split: train_full + test ----
        train_full_indices, test_indices, train_full_labels, test_labels = train_test_split(
            range(len(transformed_data)),
            transformed_data.labels,
            stratify=transformed_data.labels,
            test_size=1 - train_ratio,
            random_state=SEED
        )

        # ---- Second split: train + val ----
        # Compute val size based on train_full size
        val_size = int(len(train_full_indices) * val_ratio)
        train_size = len(train_full_indices) - val_size

        train_indices, val_indices, _, _ = train_test_split(
            train_full_indices,
            train_full_labels,
            stratify=train_full_labels,
            test_size=val_size,
            random_state=SEED
        )

        print("Train set size:", len(train_indices))
        print("Validation set size:", len(val_indices))
        print("Test set size:", len(test_indices))

        # Build subsets
        dataset_train = Subset(transformed_data, train_indices)
        dataset_val   = Subset(transformed_data, val_indices)
        dataset_test  = Subset(transformed_data, test_indices)

        return dataset_train, dataset_val, dataset_test
    
    @staticmethod
    def get_loaderss(dataset_train, dataset_test,
                    batch_size=128, num_workers=2):

        def collate_cnn(batch):
            # batch: lista tupli (A, B, filename)
            A_batch = torch.stack([item[0].squeeze(1) if item[0].dim() == 3 else item[0] for item in batch])
            B_batch = torch.stack([item[1].squeeze(1) if item[1].dim() == 3 else item[1] for item in batch])
            filenames = [item[2] for item in batch]
            return A_batch, B_batch, filenames

        train_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=collate_cnn
        )

        test_loader = DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            collate_fn=collate_cnn
        )

        return train_loader, test_loader

    @staticmethod
    def get_loaders(dataset_train, dataset_val, dataset_test,
                    batch_size=128, num_workers=2):

        def collate_cnn(batch):
            # batch: lista tupli (A, B, filename)
            A_batch = torch.stack([item[0].squeeze(1) if item[0].dim() == 3 else item[0] for item in batch])
            B_batch = torch.stack([item[1].squeeze(1) if item[1].dim() == 3 else item[1] for item in batch])
            filenames = [item[2] for item in batch]
            return A_batch, B_batch, filenames

        train_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=collate_cnn
        )

        val_loader = DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            collate_fn=collate_cnn
        )

        test_loader = DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            collate_fn=collate_cnn
        )

        return train_loader, val_loader, test_loader


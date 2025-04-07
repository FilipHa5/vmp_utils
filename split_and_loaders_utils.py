from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms

def get_train_test_dataset(transformed_data, train_ratio=0.7):
	SEED = None

	# generate train and test indices
	train_indices, test_indices, train_labels, test_labels = train_test_split(
	    range(len(transformed_data)),
	    transformed_data.labels,
	    stratify = transformed_data.labels,
	    test_size = 1 - train_ratio,
	    random_state = SEED
	    )

	print("Train set size:", len(train_indices))
	print("Test set size:", len(test_indices))

	# generate subset based on indices
	dataset_train = Subset(transformed_data, train_indices)
	dataset_test = Subset(transformed_data, test_indices)

	return dataset_train, dataset_test

def get_tr_test_loaders(dataset_train, dataset_test, batch_size=128, num_workers=0):
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

        return train_loader, test_loader

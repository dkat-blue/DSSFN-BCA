# src/datasets.py
# Contains the PyTorch Dataset class and DataLoader creation function.

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np # Needed for type hint and checking if data exists

class HyperspectralDataset(Dataset):
    """
    Custom PyTorch Dataset for Hyperspectral Patches.
    Transposes patches from (N, H, W, B) to PyTorch's expected (N, B, H, W).
    """
    def __init__(self, patches, labels):
        """
        Args:
            patches (np.ndarray): Numpy array of patches, shape (N, H, W, B).
            labels (np.ndarray): Numpy array of corresponding labels, shape (N,).
                                 Assumed to be 0-indexed integers.
        """
        if patches is None or labels is None or len(patches) == 0 or len(labels) == 0:
            print("Warning: Empty patches or labels received in HyperspectralDataset. Dataset will be empty.")
            self.patches = torch.empty((0, 1, 1, 1), dtype=torch.float32) # Placeholder empty tensor
            self.labels = torch.empty((0,), dtype=torch.long)
        elif patches.shape[0] != labels.shape[0]:
             raise ValueError(f"Number of patches ({patches.shape[0]}) must match number of labels ({labels.shape[0]})")
        else:
            # Data needs to be in (N, C, H, W) for PyTorch Conv2D
            # Input patches are (N, H, W, B=C), transpose dimensions 1, 2, 3 -> 3, 1, 2
            self.patches = torch.from_numpy(patches.transpose(0, 3, 1, 2)).float()
            self.labels = torch.from_numpy(labels).long() # Ensure labels are Long type for CrossEntropyLoss

        # Debug print shape
        # print(f"Dataset created. Patches shape: {self.patches.shape}, Labels shape: {self.labels.shape}")


    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Returns the patch and label at the given index."""
        if idx >= len(self.labels):
            raise IndexError("Index out of bounds")
        return self.patches[idx], self.labels[idx]


def create_dataloaders(data_splits, batch_size, num_workers=0, pin_memory=False):
    """
    Creates PyTorch DataLoaders for train, validation, and test sets.

    Args:
        data_splits (dict): Dictionary containing keys like 'train_patches', 'train_labels',
                            'val_patches', 'val_labels', 'test_patches', 'test_labels'.
                            Values should be numpy arrays from patch creation.
        batch_size (int): The batch size for the DataLoaders.
        num_workers (int): Number of subprocesses to use for data loading. Defaults to 0.
        pin_memory (bool): If True, copies Tensors into CUDA pinned memory before returning them.
                           Defaults to False.

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders.
              Returns None for a loader if corresponding data is missing/empty in data_splits.
    """
    loaders = {}

    # Create Train Loader
    if 'train_patches' in data_splits and 'train_labels' in data_splits and \
       data_splits['train_patches'] is not None and data_splits['train_labels'] is not None and \
       len(data_splits['train_patches']) > 0:
        train_dataset = HyperspectralDataset(data_splits['train_patches'], data_splits['train_labels'])
        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True, # Shuffle training data
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False # Keep partial batch at the end if needed
        )
        print(f"Train Loader created: {len(loaders['train'])} batches.")
    else:
        print("Train patches/labels missing or empty. Train Loader not created.")
        loaders['train'] = None

    # Create Validation Loader
    if 'val_patches' in data_splits and 'val_labels' in data_splits and \
       data_splits['val_patches'] is not None and data_splits['val_labels'] is not None and \
       len(data_splits['val_patches']) > 0:
        val_dataset = HyperspectralDataset(data_splits['val_patches'], data_splits['val_labels'])
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False, # No need to shuffle validation data
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        print(f"Validation Loader created: {len(loaders['val'])} batches.")
    else:
        print("Validation patches/labels missing or empty. Validation Loader not created.")
        loaders['val'] = None

    # Create Test Loader
    if 'test_patches' in data_splits and 'test_labels' in data_splits and \
       data_splits['test_patches'] is not None and data_splits['test_labels'] is not None and \
       len(data_splits['test_patches']) > 0:
        test_dataset = HyperspectralDataset(data_splits['test_patches'], data_splits['test_labels'])
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False, # No need to shuffle test data
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        print(f"Test Loader created: {len(loaders['test'])} batches.")
    else:
        print("Test patches/labels missing or empty. Test Loader not created.")
        loaders['test'] = None

    return loaders
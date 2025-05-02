# src/data_utils.py
# Utility functions for loading, preprocessing, and preparing hyperspectral data patches.

import os
import numpy as np
from sklearn.preprocessing import minmax_scale

def load_hyperspectral_data(data_path, data_file, gt_file, expected_data_shape=None, expected_gt_shape=None):
    """
    Loads the hyperspectral data and ground truth map from npy files.

    Args:
        data_path (str): Path to the directory containing the files.
        data_file (str): Filename of the hyperspectral data cube (.npy).
        gt_file (str): Filename of the ground truth map (.npy).
        expected_data_shape (tuple, optional): Expected shape (H, W, B) for validation. Defaults to None.
        expected_gt_shape (tuple, optional): Expected shape (H, W) for validation. Defaults to None.

    Returns:
        tuple: (data_cube, gt_map)
            - data_cube (np.ndarray): Hyperspectral data (H, W, B) as float32.
            - gt_map (np.ndarray): Ground truth labels (H, W) as int32.

    Raises:
        FileNotFoundError: If data or ground truth files are not found.
    """
    data_cube_path = os.path.join(data_path, data_file)
    gt_map_path = os.path.join(data_path, gt_file)

    if not os.path.exists(data_cube_path):
        raise FileNotFoundError(f"Data file not found: {data_cube_path}")
    if not os.path.exists(gt_map_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_map_path}")

    data_cube = np.load(data_cube_path)
    gt_map = np.load(gt_map_path)

    print(f"Data cube loaded: shape={data_cube.shape}, dtype={data_cube.dtype}")
    print(f"Ground truth map loaded: shape={gt_map.shape}, dtype={gt_map.dtype}")

    # Optional basic validation based on config
    if expected_data_shape and data_cube.shape != expected_data_shape:
        print(f"Warning: Loaded data shape {data_cube.shape} differs from expected {expected_data_shape}")
    if expected_gt_shape and gt_map.shape != expected_gt_shape:
        print(f"Warning: Loaded GT shape {gt_map.shape} differs from expected {expected_gt_shape}")

    # Convert to expected types
    data_cube = data_cube.astype(np.float32)
    gt_map = gt_map.astype(np.int32)

    return data_cube, gt_map

def normalize_data(data_cube):
    """
    Normalizes the data cube to the range [0, 1] band-wise using min-max scaling.

    Args:
        data_cube (np.ndarray): Input hyperspectral data (H, W, B).

    Returns:
        np.ndarray: Normalized data cube (H, W, B) as float32.
    """
    H, W, B = data_cube.shape
    # Reshape to (H*W, B) for scaling, then back to (H, W, B)
    normalized_data = minmax_scale(data_cube.reshape(-1, B)).reshape(H, W, B)

    print(f"Data normalized (min-max scaling per band). Min: {np.min(normalized_data)}, Max: {np.max(normalized_data)}")
    return normalized_data.astype(np.float32)

def pad_data(data_cube, border_size):
    """
    Pads the data cube with mirror padding along spatial dimensions.

    Args:
        data_cube (np.ndarray): Data cube to pad (H, W, B).
        border_size (int): Number of pixels to pad on each side spatially.

    Returns:
        np.ndarray: Padded data cube (H+2*border, W+2*border, B).
    """
    if border_size <= 0:
        print("Border size is 0 or negative, returning original data.")
        return data_cube

    padded_data = np.pad(
        data_cube,
        ((border_size, border_size), (border_size, border_size), (0, 0)),
        mode='reflect' # Mirror padding as used in the notebook
    )
    print(f"Data padded with border size {border_size}. New shape: {padded_data.shape}")
    return padded_data

def create_patches_from_coords(padded_data, coords_list, patch_size):
    """
    Creates patches and extracts corresponding labels based on a list of center coordinates.

    Args:
        padded_data (np.ndarray): Padded hyperspectral data (H_padded, W_padded, B).
        coords_list (list): List of coordinate dictionaries, e.g., [{'r': r, 'c': c, 'label': label}, ...].
                              Coordinates (r, c) should be relative to the *original* image dimensions.
        patch_size (int): The spatial dimension (height and width) of the patches to extract.

    Returns:
        tuple: (patches, labels)
            - patches (np.ndarray): Array of extracted patches (N, patch_size, patch_size, B).
            - labels (np.ndarray): Array of corresponding labels (N,).
    """
    B = padded_data.shape[-1]
    patches = []
    labels = []

    if not coords_list: # Handle empty list case
        print("Warning: Coords list is empty. Returning empty patch and label arrays.")
        return np.empty((0, patch_size, patch_size, B), dtype=padded_data.dtype), \
               np.empty((0,), dtype=np.int32)

    for coord in coords_list:
        r, c = coord['r'], coord['c'] # Original image coordinates
        label = coord['label']        # Assumes 0-indexed label is present

        r_start, c_start = r, c # Assuming these are top-left indices in padded data
        r_end = r_start + patch_size
        c_end = c_start + patch_size

        # Check boundary conditions (although padding should prevent issues if coords are correct)
        if r_end > padded_data.shape[0] or c_end > padded_data.shape[1]:
             print(f"Warning: Patch for coord {coord} exceeds padded data boundaries. Skipping.")
             continue

        patch = padded_data[r_start:r_end, c_start:c_end, :]
        patches.append(patch)
        labels.append(label) # Append the 0-indexed label

    if patches:
        print(f"Created {len(patches)} patches of size {patch_size}x{patch_size}x{B}.")
    else:
         print("No patches were created.")

    return np.array(patches), np.array(labels)
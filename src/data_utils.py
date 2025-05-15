# src/data_utils.py
# Utility functions for loading, preprocessing, and preparing hyperspectral data patches.

import os
import numpy as np
from sklearn.preprocessing import minmax_scale
import scipy.io # Added to load .mat files
import logging # Added for better logging

def load_hyperspectral_data(data_path, data_file, gt_file,
                            expected_data_shape=None, expected_gt_shape=None,
                            data_mat_key=None, gt_mat_key=None):
    """
    Loads the hyperspectral data and ground truth map from .npy or .mat files.

    Args:
        data_path (str): Path to the directory containing the files.
        data_file (str): Filename of the hyperspectral data cube (.npy or .mat).
        gt_file (str): Filename of the ground truth map (.npy or .mat).
        expected_data_shape (tuple, optional): Expected shape (H, W, B) for validation. Defaults to None.
        expected_gt_shape (tuple, optional): Expected shape (H, W) for validation. Defaults to None.
        data_mat_key (str, optional): Key for the data array if loading from .mat file.
        gt_mat_key (str, optional): Key for the ground truth array if loading from .mat file.


    Returns:
        tuple: (data_cube, gt_map)
            - data_cube (np.ndarray): Hyperspectral data (H, W, B) as float32.
            - gt_map (np.ndarray): Ground truth labels (H, W) as int32.

    Raises:
        FileNotFoundError: If data or ground truth files are not found.
        ValueError: If .mat file keys are needed but not provided, or if data cannot be extracted.
    """
    data_cube_path = os.path.join(data_path, data_file)
    gt_map_path = os.path.join(data_path, gt_file)

    if not os.path.exists(data_cube_path):
        raise FileNotFoundError(f"Data file not found: {data_cube_path}")
    if not os.path.exists(gt_map_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_map_path}")

    # Load data based on file extension
    if data_file.lower().endswith('.npy'):
        logging.info(f"Loading .npy data file: {data_file}")
        data_cube = np.load(data_cube_path)
    elif data_file.lower().endswith('.mat'):
        logging.info(f"Loading .mat data file: {data_file}")
        if data_mat_key is None:
            raise ValueError(f"data_mat_key must be provided for .mat file: {data_file}")
        mat_data = scipy.io.loadmat(data_cube_path)
        if data_mat_key not in mat_data:
            raise ValueError(f"Key '{data_mat_key}' not found in .mat file: {data_file}. Available keys: {list(mat_data.keys())}")
        data_cube = mat_data[data_mat_key]
    else:
        raise ValueError(f"Unsupported data file extension: {data_file}. Must be .npy or .mat.")

    if gt_file.lower().endswith('.npy'):
        logging.info(f"Loading .npy ground truth file: {gt_file}")
        gt_map = np.load(gt_map_path)
    elif gt_file.lower().endswith('.mat'):
        logging.info(f"Loading .mat ground truth file: {gt_file}")
        if gt_mat_key is None:
            raise ValueError(f"gt_mat_key must be provided for .mat file: {gt_file}")
        mat_gt = scipy.io.loadmat(gt_map_path)
        if gt_mat_key not in mat_gt:
            raise ValueError(f"Key '{gt_mat_key}' not found in .mat file: {gt_file}. Available keys: {list(mat_gt.keys())}")
        gt_map = mat_gt[gt_mat_key]
    else:
        raise ValueError(f"Unsupported ground truth file extension: {gt_file}. Must be .npy or .mat.")

    logging.info(f"Data cube loaded: shape={data_cube.shape}, dtype={data_cube.dtype}")
    logging.info(f"Ground truth map loaded: shape={gt_map.shape}, dtype={gt_map.dtype}")

    # Optional basic validation based on config
    if expected_data_shape and data_cube.shape != expected_data_shape:
        logging.warning(f"Loaded data shape {data_cube.shape} differs from expected {expected_data_shape}")
    if expected_gt_shape and gt_map.shape != expected_gt_shape:
        logging.warning(f"Loaded GT shape {gt_map.shape} differs from expected {expected_gt_shape}")

    # Convert to expected types
    data_cube = data_cube.astype(np.float32)
    gt_map = gt_map.astype(np.int32) # Ensure GT is integer type

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
    logging.info(f"Data normalized (min-max scaling per band). Min: {np.min(normalized_data)}, Max: {np.max(normalized_data)}")
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
        logging.info("Border size is 0 or negative, returning original data.")
        return data_cube

    padded_data = np.pad(
        data_cube,
        ((border_size, border_size), (border_size, border_size), (0, 0)),
        mode='reflect'  # Mirror padding
    )
    logging.info(f"Data padded with border size {border_size}. New shape: {padded_data.shape}")
    return padded_data

def create_patches_from_coords(padded_data, coords_list, patch_size):
    """
    Creates patches and extracts corresponding labels based on a list of center coordinates.

    Args:
        padded_data (np.ndarray): Padded hyperspectral data (H_padded, W_padded, B).
        coords_list (list): List of coordinate dictionaries, e.g.,
                            [{'r': r, 'c': c, 'label': label}, ...].
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
        logging.warning("Coords list is empty. Returning empty patch and label arrays.")
        return np.empty((0, patch_size, patch_size, B), dtype=padded_data.dtype), \
               np.empty((0,), dtype=np.int32)

    border_offset = patch_size // 2 # Offset due to padding

    for coord in coords_list:
        # The patch should be centered around `(r + border_offset, c + border_offset)` in the padded image.
        # So, the top-left corner of the patch in the padded image is:
        # r_start_padded = (r + border_offset) - patch_size // 2
        # c_start_padded = (c + border_offset) - patch_size // 2

        r_original, c_original = coord['r'], coord['c']
        label = coord['label'] # Assumes 0-indexed label is present

        # Calculate top-left corner in the padded image
        r_start_in_padded = r_original # r_original + border_offset - border_offset
        c_start_in_padded = c_original # c_original + border_offset - border_offset

        # Extract the patch
        patch = padded_data[
            r_start_in_padded : r_start_in_padded + patch_size,
            c_start_in_padded : c_start_in_padded + patch_size,
            :
        ]

        if patch.shape != (patch_size, patch_size, B):
            logging.warning(f"Patch for coord {coord} has unexpected shape {patch.shape}. Expected ({patch_size}, {patch_size}, {B}). Skipping.")
            continue

        patches.append(patch)
        labels.append(label) # Append the 0-indexed label

    if patches:
        logging.info(f"Created {len(patches)} patches of size {patch_size}x{patch_size}x{B}.")
    else:
        logging.warning("No patches were created from the provided coordinates.")

    return np.array(patches, dtype=padded_data.dtype), np.array(labels, dtype=np.int32)

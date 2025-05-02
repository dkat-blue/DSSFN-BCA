# src/sampling.py
# Functions for sampling pixels and splitting data for training, validation, and testing.

import numpy as np
import random
from sklearn.model_selection import train_test_split

def get_labeled_coordinates_and_indices(gt_map):
    """
    Finds coordinates, 0-indexed labels, and original linear indices of all labeled pixels
    (pixels with gt_map value > 0).

    Args:
        gt_map (np.ndarray): Ground truth map (H, W), where 0 is background.

    Returns:
        tuple: (labeled_coords, labels_array, original_indices)
            - labeled_coords (list): List of dicts [{'r': r, 'c': c, 'label': label_0_indexed}, ...].
            - labels_array (np.ndarray): Numpy array of 0-indexed labels corresponding to labeled_coords.
            - original_indices (np.ndarray): Numpy array of linear indices corresponding to labeled_coords.
    """
    labeled_coords = []
    labels_list = []
    original_indices_list = []
    H, W = gt_map.shape
    idx_map = np.arange(H * W).reshape((H, W)) # Map (r, c) to linear index

    print("Scanning ground truth map for labeled pixels...")
    count = 0
    for r in range(H):
        for c in range(W):
            label = gt_map[r, c]
            if label > 0: # Assuming 0 is background, labels are 1 to N
                label_0_indexed = label - 1 # Convert to 0-indexed for consistency
                labeled_coords.append({'r': r, 'c': c, 'label': label_0_indexed})
                labels_list.append(label_0_indexed)
                original_indices_list.append(idx_map[r, c])
                count += 1

    print(f"Found {len(labeled_coords)} labeled pixels across {np.max(gt_map)} classes.")

    if not labeled_coords:
        print("Warning: No labeled pixels found in the ground truth map.")
        return [], np.array([]), np.array([])

    return labeled_coords, np.array(labels_list), np.array(original_indices_list)


def split_data_random_stratified(labeled_coords, labels_array, original_indices,
                                 train_ratio, val_ratio, num_classes, random_seed):
    """
    Splits data indices using stratified random sampling based on labels,
    then maps indices back to coordinates.

    Args:
        labeled_coords (list): List of dicts [{'r': r, 'c': c, 'label': label_0_indexed}, ...].
        labels_array (np.ndarray): Numpy array of 0-indexed labels corresponding to labeled_coords.
        original_indices (np.ndarray): Numpy array of linear indices corresponding to labeled_coords.
        train_ratio (float): Proportion of samples per class for training (e.g., 0.10).
        val_ratio (float): Proportion of samples per class for validation (relative to *original* data).
        num_classes (int): Total number of classes (e.g., 16 for Indian Pines).
        random_seed (int): Seed for reproducibility.

    Returns:
        dict: Dictionary containing shuffled lists of coordinate dictionaries for each set.
              {'train_coords': [...], 'val_coords': [...], 'test_coords': [...]}
              Coordinates 'r', 'c' are relative to the original image dimensions.
              Labels 'label' are 0-indexed.
    """
    if len(original_indices) == 0:
         print("Warning: Cannot split data, no labeled indices provided.")
         return {'train_coords': [], 'val_coords': [], 'test_coords': []}

    print(f"Starting stratified split: Train={train_ratio*100}%, Val={val_ratio*100}%")

    # --- Stratified Split using Indices ---
    # Calculate test ratio needed for sklearn's train_test_split
    test_ratio_from_total = 1.0 - train_ratio - val_ratio
    if test_ratio_from_total < 0:
        raise ValueError("Train ratio + Val ratio cannot exceed 1.0")

    # First split: Separate train set from the rest (val + test)
    try:
        train_indices, remaining_indices = train_test_split(
            original_indices,
            test_size=(val_ratio + test_ratio_from_total), # Size of (val + test)
            random_state=random_seed,
            stratify=labels_array # Stratify based on all original labels
        )
    except ValueError as e:
         print(f"Warning: Stratified split for train failed (possibly too few samples per class). Error: {e}. Returning empty splits.")
         # Fallback: return empty lists or handle differently
         return {'train_coords': [], 'val_coords': [], 'test_coords': list(labeled_coords)} # Or put all in test


    # Get labels corresponding to the remaining indices for the second split
    index_to_label = {idx: lab for idx, lab in zip(original_indices, labels_array)}
    remaining_labels = np.array([index_to_label[idx] for idx in remaining_indices])

    # Calculate validation ratio relative to the *remaining* data for the second split
    # relative_val_ratio = val_size / (val_size + test_size)
    if (val_ratio + test_ratio_from_total) > 1e-9: # Avoid division by zero if only train set exists
        relative_val_ratio = val_ratio / (val_ratio + test_ratio_from_total)
    else:
        relative_val_ratio = 0 # No validation or test set needed

    # Second split: Separate val and test from the remaining indices
    if len(remaining_indices) > 0 and relative_val_ratio > 0 and relative_val_ratio < 1:
        try:
            val_indices, test_indices = train_test_split(
                remaining_indices,
                test_size=(1.0 - relative_val_ratio), # Size of test relative to remaining
                random_state=random_seed + 1, # Use different seed
                stratify=remaining_labels # Stratify based on remaining labels
            )
        except ValueError as e:
            print(f"Warning: Stratified split for val/test failed. Putting remaining into test. Error: {e}")
            val_indices = np.array([], dtype=original_indices.dtype)
            test_indices = remaining_indices
    elif len(remaining_indices) > 0 and relative_val_ratio >= 1: # If val_ratio implies taking all remaining
         val_indices = remaining_indices
         test_indices = np.array([], dtype=original_indices.dtype)
         print("Validation ratio covers all remaining samples. Test set will be empty.")
    elif len(remaining_indices) > 0: # If no validation needed (val_ratio=0)
        val_indices = np.array([], dtype=original_indices.dtype)
        test_indices = remaining_indices
        print("Validation ratio is zero. Test set gets all remaining samples.")
    else: # If remaining_indices is empty
        val_indices = np.array([], dtype=original_indices.dtype)
        test_indices = np.array([], dtype=original_indices.dtype)

    print(f"Stratified split counts: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    print(f"Total labeled pixels processed: {len(train_indices) + len(val_indices) + len(test_indices)}")

    # --- Map indices back to coordinates ---
    coord_map = {idx: coord for idx, coord in zip(original_indices, labeled_coords)}

    # Use list comprehension with check for safety (shouldn't happen with split logic)
    train_coords = [coord_map[i] for i in train_indices if i in coord_map]
    val_coords = [coord_map[i] for i in val_indices if i in coord_map]
    test_coords = [coord_map[i] for i in test_indices if i in coord_map]

    # Check if lengths match (debugging)
    if len(train_coords) != len(train_indices) or \
       len(val_coords) != len(val_indices) or \
       len(test_coords) != len(test_indices):
        print("Warning: Mismatch between index count and coordinate count after mapping.")

    # Shuffle lists (optional, as DataLoader can shuffle - but done in notebook)
    random.seed(random_seed + 2)
    random.shuffle(train_coords)
    random.shuffle(val_coords)
    random.shuffle(test_coords)

    split_coords = {
        'train_coords': train_coords,
        'val_coords': val_coords,
        'test_coords': test_coords
    }

    return split_coords
# inspect_botswana_dataset.py
# A script to load and inspect the Botswana hyperspectral dataset (.mat files)
# to gather necessary information for model configuration.

import scipy.io
import numpy as np
import os

def inspect_hyperspectral_mat(data_path, data_filename, gt_filename):
    """
    Loads and inspects hyperspectral data and ground truth from .mat files.

    Args:
        data_path (str): The directory path where the .mat files are located.
        data_filename (str): The filename of the hyperspectral data .mat file.
        gt_filename (str): The filename of the ground truth .mat file.
    """
    data_filepath = os.path.join(data_path, data_filename)
    gt_filepath = os.path.join(data_path, gt_filename)

    print(f"--- Inspecting Hyperspectral Data: {data_filename} ---")
    if not os.path.exists(data_filepath):
        print(f"ERROR: Data file not found at {data_filepath}")
        return
    try:
        data_mat = scipy.io.loadmat(data_filepath)
        print(f"Keys in '{data_filename}': {list(data_mat.keys())}")

        # Try to guess the data key (often it's the filename without extension or a common key)
        data_key_guess = data_filename.split('.')[0]
        if data_key_guess in data_mat:
            data_cube = data_mat[data_key_guess]
            print(f"Using key '{data_key_guess}' for data cube.")
        elif len(data_mat.keys()) == 3 and any(k.startswith('__') for k in data_mat.keys()): # __header__, __version__, __globals__
            # If only metadata keys and one other, assume the other is the data
            data_key_actual = [k for k in data_mat.keys() if not k.startswith('__')][0]
            data_cube = data_mat[data_key_actual]
            print(f"Found potential data key '{data_key_actual}'.")
        else:
            print(f"Could not automatically identify the data array within '{data_filename}'.")
            print("Please inspect the keys above and choose the correct one for your data cube.")
            # Prompt user for the key if multiple non-standard keys exist
            if len([k for k in data_mat.keys() if not k.startswith('__')]) > 1:
                chosen_key = input(f"Enter the key for the data array from {list(data_mat.keys())}: ")
                if chosen_key in data_mat:
                    data_cube = data_mat[chosen_key]
                else:
                    print(f"Invalid key '{chosen_key}'. Exiting data inspection.")
                    return
            else:
                print("No clear data key found.")
                return


        print(f"Data cube shape (H, W, B): {data_cube.shape}")
        print(f"Data cube data type: {data_cube.dtype}")
        print(f"Number of spectral bands: {data_cube.shape[-1]}")
        print(f"Spatial dimensions (Height, Width): ({data_cube.shape[0]}, {data_cube.shape[1]})")

    except Exception as e:
        print(f"Error loading or processing '{data_filename}': {e}")
        return # Stop if data file fails

    print(f"\n--- Inspecting Ground Truth: {gt_filename} ---")
    if not os.path.exists(gt_filepath):
        print(f"ERROR: Ground truth file not found at {gt_filepath}")
        return
    try:
        gt_mat = scipy.io.loadmat(gt_filepath)
        print(f"Keys in '{gt_filename}': {list(gt_mat.keys())}")

        gt_key_guess = gt_filename.split('.')[0]
        if gt_key_guess in gt_mat:
            gt_map = gt_mat[gt_key_guess]
            print(f"Using key '{gt_key_guess}' for ground truth map.")
        elif len(gt_mat.keys()) == 3 and any(k.startswith('__') for k in gt_mat.keys()):
             gt_key_actual = [k for k in gt_mat.keys() if not k.startswith('__')][0]
             gt_map = gt_mat[gt_key_actual]
             print(f"Found potential ground truth key '{gt_key_actual}'.")
        else:
            print(f"Could not automatically identify the ground truth array within '{gt_filename}'.")
            print("Please inspect the keys above and choose the correct one for your ground truth map.")
            if len([k for k in gt_mat.keys() if not k.startswith('__')]) > 1:
                chosen_gt_key = input(f"Enter the key for the ground truth array from {list(gt_mat.keys())}: ")
                if chosen_gt_key in gt_mat:
                    gt_map = gt_mat[chosen_gt_key]
                else:
                    print(f"Invalid key '{chosen_gt_key}'. Exiting ground truth inspection.")
                    return
            else:
                print("No clear ground truth key found.")
                return


        print(f"Ground truth map shape (H, W): {gt_map.shape}")
        print(f"Ground truth map data type: {gt_map.dtype}")

        unique_classes = np.unique(gt_map)
        print(f"Unique classes in ground truth: {unique_classes}")
        # Number of classes, excluding background/zero if present
        num_actual_classes = len(unique_classes) if 0 not in unique_classes else len(unique_classes) - 1
        print(f"Number of actual classes (excluding 0 if present as background): {num_actual_classes}")

        if data_cube.shape[0] != gt_map.shape[0] or data_cube.shape[1] != gt_map.shape[1]:
            print(f"WARNING: Spatial dimensions of data cube ({data_cube.shape[0]}x{data_cube.shape[1]}) "
                  f"do not match ground truth map ({gt_map.shape[0]}x{gt_map.shape[1]}).")

    except Exception as e:
        print(f"Error loading or processing '{gt_filename}': {e}")

if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Update these paths and filenames according to where you've saved the Botswana dataset.
    # Assuming the files are in a 'data/botswana/' subdirectory relative to where you run the script.
    # Or provide an absolute path.
    DATASET_DIRECTORY = "./data/botswana"  # Example: "C:/Users/YourName/Datasets/Botswana" or "/path/to/your/data/botswana"
    DATA_FILENAME = "Botswana.mat"
    GT_FILENAME = "Botswana_gt.mat"
    # --- End Configuration ---

    if not os.path.isdir(DATASET_DIRECTORY):
        print(f"ERROR: Dataset directory not found: {os.path.abspath(DATASET_DIRECTORY)}")
        print("Please update the 'DATASET_DIRECTORY' variable in this script.")
    else:
        inspect_hyperspectral_mat(DATASET_DIRECTORY, DATA_FILENAME, GT_FILENAME)

    print("\n--- Information for config.py ---")
    print("Based on the output above, you'll need to update your 'src/config.py' for the Botswana dataset.")
    print("Specifically, you'll need to set:")
    print("  - DATA_FILE (e.g., 'Botswana.mat')")
    print("  - GT_FILE (e.g., 'Botswana_gt.mat')")
    print("  - DATA_MAT_KEY (the key for the data array, e.g., 'Botswana' or as identified above)")
    print("  - GT_MAT_KEY (the key for the ground truth array, e.g., 'Botswana_gt' or as identified above)")
    print("  - NUM_CLASSES (the number of actual classes, e.g., 14 as mentioned in the Canvas document)")
    print("  - EXPECTED_DATA_SHAPE (e.g., (1476, 256, 145) - H, W, B_after_preprocessing)")
    print("  - EXPECTED_GT_SHAPE (e.g., (1476, 256))")
    print("  - CLASS_NAMES (a list of names for the classes, if available)")
    print("Remember that the number of bands in EXPECTED_DATA_SHAPE should be the number of bands *after* any typical preprocessing like removing noisy/water absorption bands, as mentioned in the Canvas (typically 145 for Botswana). The script will show the raw number of bands from the .mat file.")


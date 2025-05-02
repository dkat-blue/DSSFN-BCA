# -*- coding: utf-8 -*-
"""
Standalone Python script for E-FDPC dc_percent Sweep Experiment.

This script systematically tests the DSSFN model performance with
bands selected by E-FDPC using varying dc_percent values.
The number of bands is determined automatically by E-FDPC for each dc_percent.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import warnings
import sys
import datetime # Added for timestamp
import json     # Added for saving config
import random   # Added for seeding
import importlib # Added for reloading modules
import time     # Added for timing
import logging  # Added for logging
import traceback # For detailed error logging
import pandas as pd # Added for results table

# --- Add src directory to path ---
# Assumes this script is in 'scripts/' and 'src/' is a sibling directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) if '__file__' in locals() else os.path.abspath(os.path.join('.', '..'))
module_path = os.path.join(project_root, 'src')

if module_path not in sys.path:
    print(f"Adding {module_path} to sys.path")
    sys.path.append(module_path)
else:
    print(f"{module_path} already in sys.path")

# --- Import Custom Modules ---
try:
    import config as cfg
    # Force reload of config in case it was changed since last import
    # Note: We will override the band selection parts of config later
    importlib.reload(cfg)
    print("Reloaded config module.")

    import data_utils as du
    import band_selection as bs
    import sampling as spl
    import datasets as ds
    from model import DSSFN
    import engine
    # Visualization might be too much for a sweep script
    # import visualization as vis
    print("Custom modules imported successfully.")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Please ensure the 'src' directory exists at: {module_path}")
    print("Check that all .py files (config.py, data_utils.py, etc.) exist in the 'src' directory.")
    raise # Stop execution if imports fail

# --- Ignore specific warnings ---
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning) # Often from matplotlib or sklearn

# ==================================
#        Sweep Parameters
# ==================================
# Define the range of dc_percent values to test for E-FDPC
# Paper suggests 1-2%. Let's test around that range.
dc_percent_sweep_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
print(f"E-FDPC dc_percent Sweep Values to test: {dc_percent_sweep_values}")

# ==================================
#        Logger Setup
# ==================================
sweep_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Create a directory for the entire sweep
sweep_output_dir = os.path.join(project_root, cfg.OUTPUT_DIR, cfg.DATASET_NAME, f"sweep_efdpc_{sweep_timestamp}")
os.makedirs(sweep_output_dir, exist_ok=True) # Create directory immediately for logging
log_file_path = os.path.join(sweep_output_dir, "sweep_efdpc_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout) # Also print to console
    ]
)

# ==================================
#        Pipeline Functions (mostly reused from main.py / sweep_bands.py)
# ==================================

def setup_environment():
    """Sets up device and deterministic settings."""
    logging.info("--- Setting up Environment ---")
    logging.info(f"Sweep output directory: {sweep_output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Settings for Deterministic Runs ---
    torch.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.RANDOM_SEED)
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logging.info("CUDA detected: Set deterministic cuDNN algorithms.")
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # Ensure consistency
        cublas_config = os.getenv('CUBLAS_WORKSPACE_CONFIG')
        if cublas_config:
            logging.info(f"CUBLAS_WORKSPACE_CONFIG set to: {cublas_config}")
        else:
            logging.warning("CUBLAS_WORKSPACE_CONFIG environment variable not detected. Deterministic CuBLAS behavior not guaranteed.")
    else:
        logging.info("CUDA not available, running on CPU.")

    logging.info(f"Set random seed: {cfg.RANDOM_SEED}")
    logging.info(f"Set cuDNN deterministic: {torch.backends.cudnn.deterministic}")
    logging.info(f"Set cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    return device

def load_data():
    """Loads the dataset."""
    logging.info("\n--- Loading Data ---")
    data_path_abs = os.path.join(project_root, cfg.DATA_PATH)
    logging.info(f"Attempting to load data from: {data_path_abs}")
    data_cube, gt_map = du.load_hyperspectral_data(
        data_path_abs, cfg.DATA_FILE, cfg.GT_FILE,
        cfg.EXPECTED_DATA_SHAPE, cfg.EXPECTED_GT_SHAPE
    )
    H, W, B_original = data_cube.shape
    logging.info(f"Original data dimensions: H={H}, W={W}, Bands={B_original}")
    logging.info(f"Number of classes (from config): {cfg.NUM_CLASSES}")
    return data_cube, gt_map, B_original

def select_bands_for_run(data_cube, dc_percent):
    """Performs band selection using E-FDPC for a specific dc_percent."""
    logging.info(f"\n--- Processing Band Selection (E-FDPC dc_percent: {dc_percent}%) ---")
    band_selection_method_run = f"E-FDPC (dc_percent={dc_percent}%)"
    selected_data = None
    selected_indices = None
    input_bands_run = data_cube.shape[-1] # Default to original if fails

    try:
        selected_data, selected_indices = bs.apply_efdpc(
            data_cube,
            dc_percent
        )
        input_bands_run = selected_data.shape[-1] # Get actual number of bands selected
        band_selection_method_run += f" -> Selected {input_bands_run} bands" # Append count to log string
        logging.info(f"Shape after E-FDPC: {selected_data.shape}")
    except Exception as e:
        logging.error(f"Error during E-FDPC for dc_percent={dc_percent}: {e}", exc_info=True)
        logging.warning("Falling back to using original bands for this run.")
        band_selection_method_run = f"E-FDPC Failed (dc={dc_percent}%) - Used Original"
        selected_data = data_cube # Fallback
        selected_indices = np.arange(data_cube.shape[-1])
        input_bands_run = data_cube.shape[-1]

    logging.info(f"Shape of 'selected_data' for this run: {selected_data.shape}")
    return selected_data, input_bands_run, band_selection_method_run

def preprocess_and_split(selected_data, gt_map, split_coords_dict):
    """Normalizes, pads, and creates patches using pre-split coordinates."""
    # Note: Takes split_coords_dict as input now
    logging.info("\n--- Normalizing Data ---")
    logging.info(f"Normalizing data with shape: {selected_data.shape}")
    normalized_data = du.normalize_data(selected_data)

    logging.info("\n--- Padding Data ---")
    padded_data = du.pad_data(normalized_data, cfg.BORDER_SIZE)

    logging.info("\n--- Creating Patches ---")
    data_splits = {}
    logging.info("Creating patches for TRAIN set...")
    data_splits['train_patches'], data_splits['train_labels'] = du.create_patches_from_coords(
        padded_data, split_coords_dict['train_coords'], cfg.PATCH_SIZE)
    logging.info(f"Created {len(data_splits.get('train_patches', []))} training patches.")

    logging.info("Creating patches for VALIDATION set...")
    data_splits['val_patches'], data_splits['val_labels'] = du.create_patches_from_coords(
        padded_data, split_coords_dict['val_coords'], cfg.PATCH_SIZE)
    logging.info(f"Created {len(data_splits.get('val_patches', []))} validation patches.")

    logging.info("Creating patches for TEST set...")
    data_splits['test_patches'], data_splits['test_labels'] = du.create_patches_from_coords(
        padded_data, split_coords_dict['test_coords'], cfg.PATCH_SIZE)
    logging.info(f"Created {len(data_splits.get('test_patches', []))} test patches.")

    # Keep test coords if needed later, though not used directly in this script after patch creation
    data_splits['test_coords'] = split_coords_dict['test_coords']
    return data_splits

def prepare_dataloaders(data_splits):
    """Creates DataLoaders."""
    logging.info("\n--- Creating DataLoaders ---")
    loaders = ds.create_dataloaders(
        data_splits, cfg.BATCH_SIZE, cfg.NUM_WORKERS, cfg.PIN_MEMORY)
    train_loader = loaders.get('train')
    val_loader = loaders.get('val')
    test_loader = loaders.get('test')
    if not train_loader: logging.warning("Train loader not created.")
    if not val_loader: logging.warning("Validation loader not created.")
    if not test_loader: logging.warning("Test loader not created.")
    return train_loader, val_loader, test_loader

def setup_training(model):
    """Sets up criterion, optimizer, and scheduler."""
    # This function is reused, but optimizer/scheduler are created per run
    logging.info("\n--- Setting up training components ---")
    criterion = nn.CrossEntropyLoss()
    logging.info(f"Loss Function: CrossEntropyLoss")

    if cfg.OPTIMIZER_TYPE.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        logging.info(f"Optimizer: AdamW (LR={cfg.LEARNING_RATE}, WD={cfg.WEIGHT_DECAY})")
    elif cfg.OPTIMIZER_TYPE.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        logging.info(f"Optimizer: Adam (LR={cfg.LEARNING_RATE}, WD={cfg.WEIGHT_DECAY})")
    elif cfg.OPTIMIZER_TYPE.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=0.9, weight_decay=cfg.WEIGHT_DECAY)
        logging.info(f"Optimizer: SGD (LR={cfg.LEARNING_RATE}, Momentum=0.9, WD={cfg.WEIGHT_DECAY})")
    else:
        raise ValueError(f"Unsupported Optimizer Type in config: {cfg.OPTIMIZER_TYPE}")

    scheduler = None
    if cfg.USE_SCHEDULER:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.SCHEDULER_STEP_SIZE, gamma=cfg.SCHEDULER_GAMMA)
        logging.info(f"Using StepLR scheduler: step={cfg.SCHEDULER_STEP_SIZE}, gamma={cfg.SCHEDULER_GAMMA}")
    else:
        logging.info("Learning rate scheduler is disabled.")
    return criterion, optimizer, scheduler

def train_evaluate_model(model, criterion, optimizer, scheduler, loaders, device):
    """Trains and evaluates the model."""
    # Reused from main.py
    train_loader, val_loader, test_loader = loaders
    trained_model, history = None, None
    training_successful = False

    logging.info("\n--- Starting Training ---")
    if train_loader is None:
        logging.error("Cannot train model: Training data loader is not available.")
    else:
        model.to(device)
        trained_model, history = engine.train_model(
            model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            train_loader=train_loader, val_loader=val_loader, device=device,
            epochs=cfg.EPOCHS, loss_epsilon=cfg.LOSS_EPSILON, use_scheduler=cfg.USE_SCHEDULER
        )
        training_successful = True
        logging.info("Training finished.")

        # Saving best model for each run is optional - can create many files
        # Consider enabling if needed, but be mindful of disk space
        # if cfg.SAVE_BEST_MODEL and val_loader and trained_model:
        #     run_specific_dir = os.path.join(sweep_output_dir, f"dc_percent_{dc_percent_log}") # Need dc_percent_log here
        #     os.makedirs(run_specific_dir, exist_ok=True)
        #     best_model_path = os.path.join(run_specific_dir, f"{cfg.DATASET_NAME}_best_model.pth")
        #     trained_model.cpu()
        #     torch.save(trained_model.state_dict(), best_model_path)
        #     logging.info(f"Best model weights saved to: {best_model_path}")
        #     trained_model.to(device) # Move back if needed

    logging.info("\n--- Evaluating Model on Test Set ---")
    oa, aa, kappa, report, test_preds, test_labels = None, None, None, None, None, None
    evaluation_successful = False
    if trained_model is not None and test_loader is not None:
        trained_model.to(device)
        oa, aa, kappa, report, test_preds, test_labels = engine.evaluate_model(
            model=trained_model, test_loader=test_loader, device=device,
            criterion=criterion, loss_epsilon=cfg.LOSS_EPSILON
        )
        evaluation_successful = True
    elif trained_model is None:
        logging.warning("Model was not trained successfully. Skipping evaluation.")
    else:
        logging.warning("Test data loader is not available. Skipping evaluation.")

    return trained_model, history, training_successful, evaluation_successful, oa, aa, kappa, report, test_preds, test_labels


# ==================================
#          Main Execution
# ==================================
if __name__ == "__main__":
    logging.info("========== Starting E-FDPC dc_percent Sweep Run ==========")
    logging.info(f"Script: {__file__ if '__file__' in locals() else 'interactive'}")
    logging.info(f"Full command: {' '.join(sys.argv)}")
    logging.info(f"Sweep Timestamp: {sweep_timestamp}")

    overall_start_time = time.time()
    results_list = [] # Initialize list to store results

    try:
        # --- Setup ---
        device = setup_environment()

        # --- Load Data (Once) ---
        data_cube, gt_map, B_original = load_data()

        # --- Split Data Indices (Once) ---
        logging.info("\n--- Splitting Data Indices (Once) ---")
        all_labeled_coords, labels_np_array, original_idx_array = \
            spl.get_labeled_coordinates_and_indices(gt_map)
        split_coords_dict = spl.split_data_random_stratified(
            all_labeled_coords, labels_np_array, original_idx_array,
            cfg.TRAIN_RATIO, cfg.VAL_RATIO, cfg.NUM_CLASSES, cfg.RANDOM_SEED
        )
        logging.info("Train/Val/Test coordinates generated.")

        # --- E-FDPC dc_percent Sweep Loop ---
        for i, dc_percent in enumerate(dc_percent_sweep_values):
            run_start_time = time.time()
            dc_percent_log = f"{dc_percent:.2f}" # Format for logging/filenames
            logging.info(f"\n===== [{i+1}/{len(dc_percent_sweep_values)}] Running for dc_percent = {dc_percent_log}% =====")

            # --- Band Selection (E-FDPC) ---
            selected_data, input_bands_run, band_selection_method_run = select_bands_for_run(data_cube, dc_percent)

            # --- Preprocessing & Patch Creation (using same split coords) ---
            data_splits = preprocess_and_split(selected_data, gt_map, split_coords_dict)

            # --- DataLoaders ---
            loaders = prepare_dataloaders(data_splits)
            train_loader, val_loader, test_loader = loaders
            if not train_loader or not test_loader: # Need train and test at minimum
                 logging.error(f"Missing train or test loader for dc_percent={dc_percent_log}%. Skipping run.")
                 continue

            # --- Model ---
            logging.info("\n--- Instantiating the DSSFN model ---")
            logging.info(f"Model input bands: {input_bands_run}")
            # Check if input_bands_run is valid (greater than 0)
            if input_bands_run <= 0:
                logging.error(f"Invalid number of input bands ({input_bands_run}) after E-FDPC for dc_percent={dc_percent_log}%. Skipping model instantiation and training.")
                # Store placeholder results indicating failure
                results_list.append({
                    "DC Percent (%)": dc_percent,
                    "Input Bands": input_bands_run, # Store the invalid band count
                    "Method": band_selection_method_run,
                    "OA": None, "AA": None, "Kappa": None,
                    "Run Time (s)": time.time() - run_start_time,
                    "Training Status": "Skipped (Invalid Bands)",
                    "Evaluation Status": "Skipped"
                })
                continue # Skip to the next dc_percent value

            model = DSSFN(
                input_bands=input_bands_run, num_classes=cfg.NUM_CLASSES, patch_size=cfg.PATCH_SIZE,
                spec_channels=cfg.SPEC_CHANNELS, spatial_channels=cfg.SPATIAL_CHANNELS
            ).to(device)
            logging.info(f"DSSFN model instantiated on device: {device}")

            # --- Training Setup ---
            criterion, optimizer, scheduler = setup_training(model)

            # --- Train & Evaluate ---
            trained_model, history, training_successful, evaluation_successful, \
            oa, aa, kappa, report, test_preds, test_labels = train_evaluate_model(
                model, criterion, optimizer, scheduler, loaders, device
            )

            # --- Timing ---
            run_end_time = time.time()
            run_duration_seconds = run_end_time - run_start_time
            logging.info(f"\nRun time for dc_percent={dc_percent_log}%: {run_duration_seconds:.2f} seconds")

            # --- Store Results ---
            results_list.append({
                "DC Percent (%)": dc_percent,
                "Input Bands": input_bands_run, # Store the number of bands selected
                "Method": band_selection_method_run, # Store the full method string
                "OA": oa,
                "AA": aa,
                "Kappa": kappa,
                "Run Time (s)": run_duration_seconds,
                "Training Status": "Success" if training_successful else "Failed/Skipped",
                "Evaluation Status": "Success" if evaluation_successful else "Failed/Skipped"
            })

            # Clean up memory
            del model, optimizer, criterion, scheduler, trained_model, history, data_splits, loaders
            if 'cuda' in str(device):
                torch.cuda.empty_cache()
            logging.info(f"Finished run for dc_percent = {dc_percent_log}%")
            # --- End of Loop Iteration ---

        # --- Sweep Finished ---
        logging.info("\n===== E-FDPC dc_percent Sweep Finished =====")

        # --- Display and Save Results Table ---
        logging.info("\n--- Sweep Results Summary ---")
        if results_list:
            results_df = pd.DataFrame(results_list)
            # Format float columns
            float_cols = ["OA", "AA", "Kappa", "Run Time (s)"]
            for col in float_cols:
                if col in results_df.columns:
                    results_df[col] = results_df[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

            # Log the results table to the log file/console
            logging.info("Results Table:\n" + results_df.to_string())

            # Save Results Table to CSV
            results_csv_path = os.path.join(sweep_output_dir, f"sweep_efdpc_results_{sweep_timestamp}.csv")
            try:
                results_df.to_csv(results_csv_path, index=False)
                logging.info(f"Results table saved to: {results_csv_path}")
            except Exception as e:
                logging.error(f"Error saving results table: {e}", exc_info=True)
        else:
            logging.warning("No results were collected during the sweep.")


        overall_end_time = time.time()
        overall_duration_seconds = overall_end_time - overall_start_time
        logging.info(f"\nTotal sweep execution time: {overall_duration_seconds:.2f} seconds")
        logging.info("========== E-FDPC Sweep Run Finished Successfully ==========")

    except Exception as e:
        logging.error(f"An error occurred during the main sweep execution: {e}")
        logging.error(traceback.format_exc()) # Log the full traceback
        logging.info("========== E-FDPC Sweep Run Finished with Errors ==========")
        sys.exit(1) # Exit with error code


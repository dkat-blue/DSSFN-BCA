# -*- coding: utf-8 -*-
"""
Standalone Python script for Combination Sweep Experiment with Cross-Validation.

This script systematically tests the DSSFN model performance across a grid of:
- Band Selection Methods (SWGMF(30), E-FDPC(2.5), None)
- Intermediate Attention (After Stage 1 vs. None)
- Final Fusion Mechanism (CrossAttention vs. AdaptiveWeight)

It performs K-fold cross-validation by running the sweep K times, each time
using a different random seed for stratified data splitting (10% train).
Results are logged per run and aggregated (mean/std) across folds.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import warnings
import sys
import datetime
import json
import random
import importlib
import time
import logging
import traceback
import pandas as pd
import itertools # For generating combinations
import copy    # For deep copying config/model states if needed
import types   # To check for module type

# --- Add src directory to path ---
# Assumes this script is in 'scripts/' and 'src/' is a sibling directory.
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
project_root = os.path.abspath(os.path.join(script_dir, '..'))
module_path = os.path.join(project_root, 'src')

if module_path not in sys.path:
    print(f"Adding {module_path} to sys.path")
    sys.path.insert(0, module_path) # Use insert(0, ...) to prioritize this path
else:
    print(f"{module_path} already in sys.path")

# --- Import Custom Modules ---
try:
    # Import config first and reload
    import src.config as base_cfg_module # Import config module
    importlib.reload(base_cfg_module)
    print("Reloaded base config module.")

    import src.data_utils as du
    import src.band_selection as bs
    import src.sampling as spl
    import src.datasets as ds
    from src.model import DSSFN
    import src.engine as engine
    # Visualization not typically needed for sweeps
    # import src.visualization as vis
    print("Custom modules imported successfully using 'src.' prefix.")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Please ensure the 'src' directory exists at: {module_path}")
    print(f"Check that '__init__.py' exists in 'src/'")
    print(f"Current sys.path: {sys.path}")
    raise # Stop execution if imports fail

# --- Ignore specific warnings ---
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ==================================
# Helper Function to Extract Config Values
# ==================================
def config_to_dict(cfg_module):
    """Extracts primitive config values from a module into a dictionary."""
    cfg_dict = {}
    for key in dir(cfg_module):
        # Skip private attributes, built-ins, modules, functions
        if not key.startswith("__") and \
           not isinstance(getattr(cfg_module, key), (types.ModuleType, types.FunctionType, types.BuiltinFunctionType)):
            value = getattr(cfg_module, key)
            # Only copy basic types that are easily copyable/serializable
            if isinstance(value, (int, float, str, list, dict, bool, type(None), tuple)):
                 # Use deepcopy for lists/dicts within the config to ensure independence
                 # Tuples are immutable, so direct assignment is fine, but deepcopy handles nested structures if needed.
                 cfg_dict[key] = copy.deepcopy(value)
            elif isinstance(value, np.ndarray):
                 cfg_dict[key] = value.copy() # Copy numpy arrays
            # Add other types if needed, but be cautious about pickling issues
    return cfg_dict

# ==================================
#        Sweep Parameters
# ==================================
# Define the grid of parameters to test
band_selection_options = [
    {'method': 'SWGMF', 'param': 30},    # SWGMF with 30 target bands
    {'method': 'E-FDPC', 'param': 2.5},  # E-FDPC with dc_percent = 2.5
    {'method': 'None', 'param': None}    # No band selection
]
intermediate_attention_options = [
    [],    # No intermediate attention
    [1]    # Intermediate attention after stage 1
]
fusion_mechanism_options = [
    'CrossAttention',
    'AdaptiveWeight'
]

# Generate all combinations
parameter_combinations = list(itertools.product(
    band_selection_options,
    intermediate_attention_options,
    fusion_mechanism_options
))
num_combinations = len(parameter_combinations)
print(f"Total parameter combinations to test: {num_combinations}")

# Cross-Validation Settings
K_FOLDS = 10 # Number of times to run with different 10% training splits
# Extract base seed from the module before converting to dict
BASE_RANDOM_SEED = base_cfg_module.RANDOM_SEED

# Define Train/Val/Test Ratios for each fold's split
FOLD_TRAIN_RATIO = 0.10
FOLD_VAL_RATIO = 0.10 # The remaining 80% will be test set for each split

# ==================================
#        Logger Setup
# ==================================
sweep_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Create a directory for the entire sweep
# Use values from the module for initial setup
sweep_output_dir = os.path.join(project_root, base_cfg_module.OUTPUT_DIR, base_cfg_module.DATASET_NAME, f"sweep_combinations_{sweep_timestamp}")
os.makedirs(sweep_output_dir, exist_ok=True) # Create directory immediately for logging
log_file_path = os.path.join(sweep_output_dir, "sweep_combinations_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s', # Include level name
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout) # Also print to console
    ]
)

# ==================================
# Pipeline Functions (Adapted to use config dictionary)
# ==================================

def setup_environment(seed):
    """Sets up device and deterministic settings FOR A SPECIFIC SEED."""
    # (No changes needed here as it doesn't use config values directly)
    logging.info(f"--- Setting up Environment (Seed: {seed}) ---")
    logging.info(f"Sweep output directory: {sweep_output_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logging.info("CUDA detected: Set deterministic cuDNN algorithms.")
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        cublas_config = os.getenv('CUBLAS_WORKSPACE_CONFIG')
        if cublas_config: logging.info(f"CUBLAS_WORKSPACE_CONFIG set to: {cublas_config}")
        else: logging.warning("CUBLAS_WORKSPACE_CONFIG not detected/set.")
    else:
        logging.info("CUDA not available, running on CPU.")
    logging.info(f"Set random seed: {seed}")
    logging.info(f"Set cuDNN deterministic: {torch.backends.cudnn.deterministic if torch.cuda.is_available() else 'N/A'}")
    logging.info(f"Set cuDNN benchmark: {torch.backends.cudnn.benchmark if torch.cuda.is_available() else 'N/A'}")
    return device

def load_data(cfg_dict): # Accepts config dictionary
    """Loads the dataset based on the provided config dictionary."""
    logging.info("\n--- Loading Data ---")
    data_path_abs = cfg_dict['DATA_PATH'] # Access via dictionary key
    logging.info(f"Attempting to load data from: {data_path_abs}")
    data_cube, gt_map = du.load_hyperspectral_data(
        data_path_abs, cfg_dict['DATA_FILE'], cfg_dict['GT_FILE'],
        cfg_dict['EXPECTED_DATA_SHAPE'], cfg_dict['EXPECTED_GT_SHAPE'] # Access keys
    )
    H, W, B_original = data_cube.shape
    logging.info(f"Original data dimensions: H={H}, W={W}, Bands={B_original}")
    logging.info(f"Number of classes (from config): {cfg_dict['NUM_CLASSES']}")
    return data_cube, gt_map, B_original

def select_bands_for_run(data_cube, B_original, band_option, cfg_dict): # Accepts config dictionary
    """Performs band selection based on the band_option for the current run."""
    method = band_option['method']
    param = band_option['param']
    logging.info(f"\n--- Processing Band Selection (Method: {method}, Param: {param}) ---")

    selected_data = data_cube
    input_bands_run = B_original
    selected_indices = np.arange(B_original)
    band_selection_method_run = f"{method}"

    if method == 'SWGMF':
        target_bands = param
        band_selection_method_run += f"(target={target_bands})"
        if target_bands is None or not isinstance(target_bands, int) or target_bands <= 0 or target_bands >= B_original:
            logging.warning(f">>> Skipping SWGMF: Invalid target_bands ({target_bands}). Using original.")
            band_selection_method_run += " - Skipped (Invalid Target)"
        else:
            # Access SWGMF_WINDOW_SIZE from dict
            logging.info(f">>> Applying SWGMF (target={target_bands}, window={cfg_dict['SWGMF_WINDOW_SIZE']})")
            try:
                selected_data, selected_indices = bs.apply_swgmf(data_cube, cfg_dict['SWGMF_WINDOW_SIZE'], target_bands)
                input_bands_run = selected_data.shape[-1]
                logging.info(f"Shape after SWGMF: {selected_data.shape}")
                if input_bands_run != target_bands:
                    logging.warning(f"SWGMF returned {input_bands_run} bands, target was {target_bands}. Using {input_bands_run}.")
                    band_selection_method_run += f" -> Found {input_bands_run}"
            except Exception as e:
                logging.error(f"Error during SWGMF: {e}", exc_info=True)
                logging.warning("Falling back to original bands.")
                band_selection_method_run += " - Failed"
                selected_data = data_cube
                input_bands_run = B_original
                selected_indices = np.arange(B_original)
    elif method == 'E-FDPC':
        dc_percent = param
        band_selection_method_run += f"(dc%={dc_percent})"
        logging.info(f">>> Applying E-FDPC (dc_percent={dc_percent})")
        try:
            selected_data, selected_indices = bs.apply_efdpc(data_cube, dc_percent)
            input_bands_run = selected_data.shape[-1]
            logging.info(f"Shape after E-FDPC: {selected_data.shape}")
            band_selection_method_run += f" -> Selected {input_bands_run}"
            if input_bands_run <= 0:
                logging.error("E-FDPC selected 0 bands. Falling back to original.")
                band_selection_method_run += " - Failed (0 bands)"
                selected_data = data_cube
                input_bands_run = B_original
                selected_indices = np.arange(B_original)
        except Exception as e:
            logging.error(f"Error during E-FDPC: {e}", exc_info=True)
            logging.warning("Falling back to original bands.")
            band_selection_method_run += " - Failed"
            selected_data = data_cube
            input_bands_run = B_original
            selected_indices = np.arange(B_original)
    elif method == 'None':
        logging.info(">>> Skipping band selection (Method is None).")
        band_selection_method_run = "None"
    else:
        logging.warning(f">>> Unknown band selection method '{method}'. Using original.")
        band_selection_method_run = f"Unknown ({method})"

    logging.info(f"Band Selection Summary: {band_selection_method_run}")
    logging.info(f"Shape of 'selected_data' for this run: {selected_data.shape}")
    return selected_data, input_bands_run, band_selection_method_run

def preprocess_and_split(selected_data, gt_map, split_coords_dict, cfg_dict): # Accepts config dictionary
    """Normalizes, pads, and creates patches using the split for the current fold."""
    logging.info("\n--- Normalizing and Padding ---")
    logging.info(f"Input shape for normalization: {selected_data.shape}")
    normalized_data = du.normalize_data(selected_data)
    # Access BORDER_SIZE from dict
    padded_data = du.pad_data(normalized_data, cfg_dict['BORDER_SIZE'])
    logging.info(f"Padded shape: {padded_data.shape}")

    logging.info("\n--- Creating Patches ---")
    data_splits = {}
    for split_name in ['train', 'val', 'test']:
        coords_key = f'{split_name}_coords'
        if coords_key in split_coords_dict and split_coords_dict[coords_key]:
            logging.info(f"Creating patches for {split_name.upper()} set ({len(split_coords_dict[coords_key])} coords)...")
            # Access PATCH_SIZE from dict
            data_splits[f'{split_name}_patches'], data_splits[f'{split_name}_labels'] = du.create_patches_from_coords(
                padded_data, split_coords_dict[coords_key], cfg_dict['PATCH_SIZE'])
            logging.info(f"  -> Created {len(data_splits.get(f'{split_name}_patches', []))} patches.")
        else:
            logging.warning(f"No coordinates found for {split_name.upper()} set. Patches will be empty.")
            data_splits[f'{split_name}_patches'], data_splits[f'{split_name}_labels'] = np.array([]), np.array([])

    data_splits['test_coords'] = split_coords_dict.get('test_coords', [])
    return data_splits

def prepare_dataloaders(data_splits, cfg_dict): # Accepts config dictionary
    """Creates DataLoaders."""
    logging.info("\n--- Creating DataLoaders ---")
    # Access BATCH_SIZE, NUM_WORKERS, PIN_MEMORY from dict
    loaders = ds.create_dataloaders(
        data_splits, cfg_dict['BATCH_SIZE'], cfg_dict['NUM_WORKERS'], cfg_dict['PIN_MEMORY'])
    train_loader = loaders.get('train')
    val_loader = loaders.get('val')
    test_loader = loaders.get('test')
    if not train_loader: logging.warning("Train loader not created or is empty.")
    if not val_loader: logging.warning("Validation loader not created or is empty.")
    if not test_loader: logging.warning("Test loader not created or is empty.")
    return train_loader, val_loader, test_loader

def setup_training(model, cfg_dict): # Accepts config dictionary
    """Sets up criterion, optimizer, and scheduler based on config dict."""
    logging.info("\n--- Setting up training components ---")
    criterion = nn.CrossEntropyLoss()
    logging.info(f"Loss Function: CrossEntropyLoss")

    # Access optimizer params from dict
    optimizer_type = cfg_dict['OPTIMIZER_TYPE']
    lr = cfg_dict['LEARNING_RATE']
    wd = cfg_dict['WEIGHT_DECAY']

    if optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        logging.info(f"Optimizer: AdamW (LR={lr}, WD={wd})")
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        logging.info(f"Optimizer: Adam (LR={lr}, WD={wd})")
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        logging.info(f"Optimizer: SGD (LR={lr}, Momentum=0.9, WD={wd})")
    else:
        raise ValueError(f"Unsupported Optimizer Type in config: {optimizer_type}")

    scheduler = None
    if cfg_dict['USE_SCHEDULER']:
        # Access scheduler params from dict
        step_size = cfg_dict['SCHEDULER_STEP_SIZE']
        gamma = cfg_dict['SCHEDULER_GAMMA']
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        logging.info(f"Using StepLR scheduler: step={step_size}, gamma={gamma}")
    else:
        logging.info("Learning rate scheduler is disabled.")
    return criterion, optimizer, scheduler

def train_evaluate_model(model, criterion, optimizer, scheduler, loaders, device, cfg_dict): # Accepts config dictionary
    """Trains and evaluates the model using the engine."""
    train_loader, val_loader, test_loader = loaders
    trained_model, history = None, None
    training_successful = False

    logging.info("\n--- Starting Training ---")
    if train_loader is None or len(train_loader.dataset) == 0:
        logging.error("Cannot train model: Training data loader is not available or empty.")
    else:
        model.to(device)
        # Access training params from dict
        trained_model, history = engine.train_model(
            model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            train_loader=train_loader, val_loader=val_loader, device=device,
            epochs=cfg_dict['EPOCHS'], loss_epsilon=cfg_dict['LOSS_EPSILON'],
            use_scheduler=cfg_dict['USE_SCHEDULER']
        )
        training_successful = True
        logging.info("Training finished.")

    logging.info("\n--- Evaluating Model on Test Set ---")
    oa, aa, kappa, report, test_preds, test_labels = None, None, None, None, None, None
    evaluation_successful = False
    if trained_model is not None and test_loader is not None and len(test_loader.dataset) > 0:
        trained_model.to(device)
        # Access evaluation params from dict
        oa, aa, kappa, report, test_preds, test_labels = engine.evaluate_model(
            model=trained_model, test_loader=test_loader, device=device,
            criterion=criterion, loss_epsilon=cfg_dict['LOSS_EPSILON']
        )
        evaluation_successful = True
    elif trained_model is None:
        logging.warning("Model was not trained successfully. Skipping evaluation.")
    else:
        logging.warning("Test data loader is not available or empty. Skipping evaluation.")

    return trained_model, history, training_successful, evaluation_successful, oa, aa, kappa, report

def format_results_table(df_agg):
    """Formats the aggregated results DataFrame into a string table."""
    # (No changes needed here)
    float_cols = ['OA_mean', 'OA_std', 'AA_mean', 'AA_std', 'Kappa_mean', 'Kappa_std', 'Time_mean', 'Time_std']
    for col in float_cols:
        if col in df_agg.columns:
            df_agg[col] = df_agg[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    display_cols = [
        'Band Selection', 'Band Param', 'Int. Attention', 'Fusion', 'Successful Folds',
        'OA_mean', 'OA_std', 'AA_mean', 'AA_std', 'Kappa_mean', 'Kappa_std', 'Time_mean', 'Time_std'
    ]
    display_cols = [col for col in display_cols if col in df_agg.columns]
    df_display = df_agg[display_cols]
    return df_display.to_string(index=False)


# ==================================
#          Main Execution
# ==================================
if __name__ == "__main__":
    logging.info("========== Starting Combination Sweep Run ==========")
    logging.info(f"Script: {__file__ if '__file__' in locals() else 'interactive'}")
    logging.info(f"Full command: {' '.join(sys.argv)}")
    logging.info(f"Sweep Timestamp: {sweep_timestamp}")
    logging.info(f"Parameter Combinations: {num_combinations}")
    logging.info(f"K-Folds: {K_FOLDS}")

    overall_start_time = time.time()
    all_run_results = [] # List to store detailed results from every run

    try:
        # --- Convert Base Config Module to Dictionary (Once) ---
        base_cfg_dict = config_to_dict(base_cfg_module)
        logging.info("Base configuration values extracted to dictionary.")

        # --- Load Data (Once) ---
        # Pass the dictionary to load_data
        data_cube, gt_map, B_original = load_data(base_cfg_dict)

        # --- Get All Labeled Coords (Once) ---
        logging.info("\n--- Extracting Labeled Pixel Coordinates (Once) ---")
        # Access NUM_CLASSES from the dict for splitting
        all_labeled_coords, labels_np_array, original_idx_array = \
            spl.get_labeled_coordinates_and_indices(gt_map)
        if len(all_labeled_coords) == 0:
             raise ValueError("No labeled pixels found in the ground truth map. Cannot proceed.")

        # --- Cross-Validation Loop ---
        for fold in range(K_FOLDS):
            fold_start_time = time.time()
            current_seed = BASE_RANDOM_SEED + fold # Use a different seed for each fold
            logging.info(f"\n\n======= Starting Fold {fold+1}/{K_FOLDS} (Seed: {current_seed}) =======")

            # --- Setup Environment for Fold ---
            device = setup_environment(current_seed)

            # --- Split Data Indices for Fold ---
            logging.info(f"\n--- Splitting Data Indices for Fold {fold+1} ---")
            # Use values from base_cfg_dict for splitting parameters
            split_coords_dict = spl.split_data_random_stratified(
                all_labeled_coords, labels_np_array, original_idx_array,
                FOLD_TRAIN_RATIO, FOLD_VAL_RATIO,
                base_cfg_dict['NUM_CLASSES'], # Get NUM_CLASSES from dict
                current_seed # Use fold-specific seed
            )
            logging.info(f"Fold {fold+1}: Train={len(split_coords_dict.get('train_coords',[]))}, Val={len(split_coords_dict.get('val_coords',[]))}, Test={len(split_coords_dict.get('test_coords',[]))}")

            # --- Parameter Combination Loop ---
            for combo_idx, (band_option, intermediate_attention, fusion_mechanism) in enumerate(parameter_combinations):
                run_start_time = time.time()
                logging.info(f"\n--- Fold {fold+1}, Combination {combo_idx+1}/{num_combinations} ---")
                logging.info(f"  Band Sel: {band_option}")
                logging.info(f"  Int Attn: {'Yes (Stage 1)' if intermediate_attention else 'No'}")
                logging.info(f"  Fusion: {fusion_mechanism}")

                # --- Create and Modify Config Dict for this Run ---
                # Start with a copy of the base dictionary
                run_cfg_dict = base_cfg_dict.copy()
                # Update with run-specific parameters
                run_cfg_dict['RANDOM_SEED'] = current_seed
                run_cfg_dict['INTERMEDIATE_ATTENTION_STAGES'] = intermediate_attention
                run_cfg_dict['FUSION_MECHANISM'] = fusion_mechanism
                run_cfg_dict['TRAIN_RATIO'] = FOLD_TRAIN_RATIO
                run_cfg_dict['VAL_RATIO'] = FOLD_VAL_RATIO
                # Note: Other base parameters like LR, EPOCHS etc. are already in run_cfg_dict

                # --- Band Selection ---
                # Pass the run-specific config dict
                selected_data, input_bands_run, band_selection_method_run = select_bands_for_run(
                    data_cube, B_original, band_option, run_cfg_dict
                )
                if input_bands_run <= 0:
                    logging.error(f"Band selection resulted in {input_bands_run} bands. Skipping this combination for Fold {fold+1}.")
                    run_status = "Skipped (Invalid Bands)"
                    oa, aa, kappa = None, None, None
                    run_duration_seconds = 0
                else:
                    # --- Preprocessing & Patch Creation ---
                    # Pass the run-specific config dict
                    data_splits = preprocess_and_split(selected_data, gt_map, split_coords_dict, run_cfg_dict)

                    # --- DataLoaders ---
                    # Pass the run-specific config dict
                    loaders = prepare_dataloaders(data_splits, run_cfg_dict)
                    train_loader, val_loader, test_loader = loaders
                    if not train_loader or len(train_loader.dataset) == 0 or not test_loader or len(test_loader.dataset) == 0:
                        logging.error(f"Missing or empty train/test loader for Fold {fold+1}, Combo {combo_idx+1}. Skipping run.")
                        run_status = "Skipped (DataLoader Error)"
                        oa, aa, kappa = None, None, None
                        run_duration_seconds = 0
                    else:
                        # --- Model Instantiation ---
                        logging.info("\n--- Instantiating the DSSFN model ---")
                        # Model instantiation uses parameters from the run_cfg_dict
                        # Need to ensure DSSFN uses these values correctly
                        # Assuming DSSFN's __init__ primarily uses explicit args,
                        # but it might read from cfg internally for INTERMEDIATE_ATTENTION_STAGES.
                        # Let's pass them explicitly for safety if DSSFN doesn't read from cfg module.
                        # *** Check DSSFN model code - it reads INTERMEDIATE_ATTENTION_STAGES from cfg module ***
                        # *** We need to temporarily modify the imported cfg module or refactor DSSFN ***
                        # *** Quick Fix: Modify the imported module's attribute temporarily ***
                        original_int_attn = base_cfg_module.INTERMEDIATE_ATTENTION_STAGES
                        original_fusion = base_cfg_module.FUSION_MECHANISM
                        base_cfg_module.INTERMEDIATE_ATTENTION_STAGES = run_cfg_dict['INTERMEDIATE_ATTENTION_STAGES']
                        base_cfg_module.FUSION_MECHANISM = run_cfg_dict['FUSION_MECHANISM']
                        # --- Instantiate Model ---
                        model = DSSFN(
                            input_bands=input_bands_run,
                            num_classes=run_cfg_dict['NUM_CLASSES'],
                            patch_size=run_cfg_dict['PATCH_SIZE'],
                            spec_channels=run_cfg_dict['SPEC_CHANNELS'],
                            spatial_channels=run_cfg_dict['SPATIAL_CHANNELS'],
                            fusion_mechanism=run_cfg_dict['FUSION_MECHANISM'], # Explicitly pass fusion
                            cross_attention_heads=run_cfg_dict['CROSS_ATTENTION_HEADS'],
                            cross_attention_dropout=run_cfg_dict['CROSS_ATTENTION_DROPOUT']
                        ).to(device)
                        # --- Restore original cfg module values ---
                        base_cfg_module.INTERMEDIATE_ATTENTION_STAGES = original_int_attn
                        base_cfg_module.FUSION_MECHANISM = original_fusion
                        logging.info(f"Model instantiated with IntAttn={model.intermediate_stages}, Fusion={model.fusion_mechanism}")


                        # --- Training Setup ---
                        # Pass the run-specific config dict
                        criterion, optimizer, scheduler = setup_training(model, run_cfg_dict)

                        # --- Train & Evaluate ---
                        # Pass the run-specific config dict
                        _, _, training_successful, evaluation_successful, \
                        oa, aa, kappa, _ = train_evaluate_model(
                            model, criterion, optimizer, scheduler, loaders, device, run_cfg_dict
                        )
                        run_status = f"Train:{'OK' if training_successful else 'Fail'}, Eval:{'OK' if evaluation_successful else 'Fail'}"

                        # --- Timing ---
                        run_end_time = time.time()
                        run_duration_seconds = run_end_time - run_start_time

                # --- Log Run Time and Results ---
                logging.info(f"Run Time (Fold {fold+1}, Combo {combo_idx+1}): {run_duration_seconds:.2f} seconds")
                logging.info(f"Result: OA={f'{oa:.4f}' if oa is not None else 'N/A'}, AA={f'{aa:.4f}' if aa is not None else 'N/A'}, Kappa={f'{kappa:.4f}' if kappa is not None else 'N/A'}")

                # --- Store Detailed Results ---
                all_run_results.append({
                    "Fold": fold + 1,
                    "Seed": current_seed,
                    "Combination Index": combo_idx + 1,
                    "Band Selection": band_option['method'],
                    "Band Param": band_option['param'], # This will be None for 'None' method
                    "Input Bands": input_bands_run,
                    "Int. Attention Stages": str(intermediate_attention), # Store as string for grouping
                    "Fusion Mechanism": fusion_mechanism,
                    "OA": oa,
                    "AA": aa,
                    "Kappa": kappa,
                    "Run Time (s)": run_duration_seconds,
                    "Status": run_status
                })

                # Clean up memory for the next combination
                if 'model' in locals(): del model
                if 'optimizer' in locals(): del optimizer
                if 'criterion' in locals(): del criterion
                if 'scheduler' in locals(): del scheduler
                if 'loaders' in locals(): del loaders
                if 'data_splits' in locals(): del data_splits
                if 'run_cfg_dict' in locals(): del run_cfg_dict # Delete the run-specific dict
                if 'cuda' in str(device):
                    torch.cuda.empty_cache()
                # --- End of Combination Loop ---

            fold_end_time = time.time()
            logging.info(f"======= Finished Fold {fold+1}/{K_FOLDS} in {fold_end_time - fold_start_time:.2f} sec =======")
            # --- End of Fold Loop ---

        # --- Sweep Finished ---
        logging.info("\n\n========== All Folds and Combinations Finished ==========")

        # --- Process and Save Results ---
        if not all_run_results:
            logging.warning("No results were collected during the sweep.")
        else:
            # Create DataFrame for detailed results
            df_detailed = pd.DataFrame(all_run_results)

            # <<< MODIFICATION START: Fill NaN in 'Band Param' before aggregation >>>
            # Replace NaN in 'Band Param' (which occurs for 'None' method) with a placeholder string
            logging.info("Filling NaN in 'Band Param' column with 'N/A' for aggregation.")
            df_detailed['Band Param'] = df_detailed['Band Param'].fillna('N/A')
            # <<< MODIFICATION END >>>

            # Save Detailed Results to CSV (with 'N/A' in Band Param for None method)
            detailed_csv_path = os.path.join(sweep_output_dir, f"sweep_combinations_detailed_results_{sweep_timestamp}.csv")
            try:
                df_detailed.to_csv(detailed_csv_path, index=False)
                logging.info(f"Detailed results table saved to: {detailed_csv_path}")
            except Exception as e:
                logging.error(f"Error saving detailed results CSV: {e}", exc_info=True)

            # Aggregate Results (Mean/Std across folds for each combination)
            logging.info("\n--- Aggregating Results Across Folds ---")
            grouping_cols = [
                'Band Selection', 'Band Param', 'Int. Attention Stages', 'Fusion Mechanism'
            ]
            # Filter out runs that were skipped or failed evaluation before aggregation
            df_valid_eval = df_detailed.dropna(subset=['OA', 'AA', 'Kappa']) # Drop rows where metrics are None/NaN

            if not df_valid_eval.empty:
                 agg_funcs = {
                     'OA': ['mean', 'std'],
                     'AA': ['mean', 'std'],
                     'Kappa': ['mean', 'std'],
                     'Run Time (s)': ['mean', 'std'],
                     'Fold': ['count'] # Count how many folds completed successfully for this combo
                 }
                 # Now grouping should work correctly as 'Band Param' contains 'N/A' instead of NaN
                 df_aggregated = df_valid_eval.groupby(grouping_cols).agg(agg_funcs)
                 df_aggregated.columns = ['_'.join(col).strip() for col in df_aggregated.columns.values]
                 df_aggregated = df_aggregated.rename(columns={'Fold_count': 'Successful Folds'})
                 df_aggregated = df_aggregated.reset_index() # Make grouping cols regular columns

                 # Rename columns for clarity
                 df_aggregated = df_aggregated.rename(columns={
                     'Int. Attention Stages': 'Int. Attention',
                     'Fusion Mechanism': 'Fusion',
                     'Run Time (s)_mean': 'Time_mean',
                     'Run Time (s)_std': 'Time_std'
                 })

                 logging.info("Aggregated Results (Mean/Std across successful folds):")
                 # Display the aggregated table in the log
                 logging.info("\n" + df_aggregated.to_string())

                 # Save Aggregated Results to CSV
                 agg_csv_path = os.path.join(sweep_output_dir, f"sweep_combinations_aggregated_results_{sweep_timestamp}.csv")
                 try:
                     df_aggregated.to_csv(agg_csv_path, index=False)
                     logging.info(f"Aggregated results table saved to: {agg_csv_path}")
                 except Exception as e:
                     logging.error(f"Error saving aggregated results CSV: {e}", exc_info=True)

                 # Save Aggregated Results to TXT Table
                 agg_txt_path = os.path.join(sweep_output_dir, f"sweep_combinations_aggregated_results_{sweep_timestamp}.txt")
                 try:
                     # Format the table for the TXT file
                     results_table_str = format_results_table(df_aggregated.copy()) # Pass a copy to avoid modifying df_aggregated
                     with open(agg_txt_path, 'w') as f:
                         f.write("Aggregated Hyperspectral Model Combination Results\n")
                         f.write("=" * 50 + "\n")
                         f.write(f"Timestamp: {sweep_timestamp}\n")
                         # Access dataset name from the original dict
                         f.write(f"Dataset: {base_cfg_dict['DATASET_NAME']}\n")
                         f.write(f"Folds Run: {K_FOLDS}\n")
                         f.write(f"Base Seed: {BASE_RANDOM_SEED}\n")
                         f.write("Metrics are Mean ± Std Dev across successful folds.\n")
                         f.write("-" * 50 + "\n\n")
                         f.write(results_table_str)
                     logging.info(f"Formatted aggregated results table saved to: {agg_txt_path}")
                 except Exception as e:
                     logging.error(f"Error saving aggregated results TXT: {e}", exc_info=True)

            else:
                 logging.warning("No runs completed evaluation successfully. Cannot aggregate results.")


        overall_end_time = time.time()
        overall_duration_seconds = overall_end_time - overall_start_time
        logging.info(f"\nTotal sweep execution time: {overall_duration_seconds:.2f} seconds ({overall_duration_seconds/3600:.2f} hours)")
        logging.info("========== Sweep Run Finished Successfully ==========")

    except Exception as e:
        logging.error(f"An error occurred during the main sweep execution: {e}")
        logging.error(traceback.format_exc()) # Log the full traceback
        logging.info("========== Sweep Run Finished with Errors ==========")
        if all_run_results:
             try:
                 df_detailed = pd.DataFrame(all_run_results)
                 # Fill NaN here too before saving partial results
                 df_detailed['Band Param'] = df_detailed['Band Param'].fillna('N/A')
                 detailed_csv_path = os.path.join(sweep_output_dir, f"sweep_combinations_DETAILED_results_ERROR_{sweep_timestamp}.csv")
                 df_detailed.to_csv(detailed_csv_path, index=False)
                 logging.info(f"Saved partial detailed results after error to: {detailed_csv_path}")
             except Exception as save_e:
                 logging.error(f"Could not save partial results after error: {save_e}")
        sys.exit(1) # Exit with error code
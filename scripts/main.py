# -*- coding: utf-8 -*-
"""
Standalone Python script for DSSFN Training and Evaluation.

This script performs the full pipeline: setup, data loading,
optional band selection, preprocessing, training, evaluation,
and saving results/logs. Allows selection between SWGMF, E-FDPC, or None
band selection, and between AdaptiveWeight or CrossAttention fusion.
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

# --- Add src directory to path ---
# <<< Corrected path calculation to ensure it finds the parent of 'scripts' >>>
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
    import src.config as cfg
    importlib.reload(cfg) # Reload config after potentially adding path
    print("Reloaded config module.")

    import src.data_utils as du
    import src.band_selection as bs
    import src.sampling as spl
    import src.datasets as ds
    from src.model import DSSFN # Import the main model class
    # Import necessary components if needed directly here (though less likely)
    # from src.modules import PyramidalResidualBlock, MultiHeadCrossAttention
    import src.engine as engine # Import the engine module
    import src.visualization as vis
    print("Custom modules imported successfully using 'src.' prefix.")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Please ensure the 'src' directory exists at: {module_path}")
    print(f"Check that '__init__.py' exists in 'src/'")
    print(f"Current sys.path: {sys.path}")
    raise


# --- Ignore specific warnings ---
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Logger Setup ---
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Ensure OUTPUT_DIR exists before creating subdirectories
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
run_output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET_NAME, f"run_{run_timestamp}")
os.makedirs(run_output_dir, exist_ok=True)
log_file_path = os.path.join(run_output_dir, "run_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

# ==================================
#        Pipeline Functions
# ==================================

def setup_environment():
    """Sets up device and deterministic settings."""
    logging.info("--- Setting up Environment ---")
    logging.info(f"Run-specific output directory: {run_output_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    torch.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.RANDOM_SEED)
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
        # Try setting deterministic algorithms
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
             # Setting for deterministic convolution algorithms (optional, might impact performance)
            # torch.use_deterministic_algorithms(True)
            logging.info("CUDA detected: Set deterministic cuDNN algorithms.")
        except Exception as e:
            logging.warning(f"Could not set deterministic CUDA settings: {e}")

        # Check for CUBLAS_WORKSPACE_CONFIG (needed for deterministic matmul on some setups)
        # It's often set outside the script, but we check if it's needed.
        # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # Example: uncomment to force set
        cublas_config = os.getenv('CUBLAS_WORKSPACE_CONFIG')
        if not cublas_config:
            logging.warning("CUBLAS_WORKSPACE_CONFIG environment variable not set. Some CUDA operations might be non-deterministic.")
        else:
             logging.info(f"CUBLAS_WORKSPACE_CONFIG is set: {cublas_config}")

    else:
        logging.info("CUDA not available, running on CPU.")

    logging.info(f"Set random seed: {cfg.RANDOM_SEED}")
    logging.info(f"Set cuDNN deterministic: {torch.backends.cudnn.deterministic if torch.cuda.is_available() else 'N/A'}")
    logging.info(f"Set cuDNN benchmark: {torch.backends.cudnn.benchmark if torch.cuda.is_available() else 'N/A'}")
    return device

def load_data():
    """Loads the dataset."""
    logging.info("\n--- Loading Data ---")
    # Use absolute path derived from config's project root calculation
    data_path_abs = cfg.DATA_PATH # config.py now calculates the absolute path
    logging.info(f"Attempting to load data from: {data_path_abs}")
    data_cube, gt_map = du.load_hyperspectral_data(
        data_path_abs, cfg.DATA_FILE, cfg.GT_FILE,
        cfg.EXPECTED_DATA_SHAPE, cfg.EXPECTED_GT_SHAPE
    )
    H, W, B_original = data_cube.shape
    logging.info(f"Original data dimensions: H={H}, W={W}, Bands={B_original}")
    logging.info(f"Number of classes (from config): {cfg.NUM_CLASSES}")
    return data_cube, gt_map, B_original

def select_bands(data_cube, B_original):
    """Performs band selection based on config."""
    logging.info("\n--- Processing Band Selection ---")
    method = cfg.BAND_SELECTION_METHOD
    band_selection_method_log = "None"
    selected_data = data_cube
    input_bands = B_original
    selected_indices = np.arange(B_original)

    logging.info(f"Configured Method: {method}")

    if method is None or str(method).upper() == 'NONE': # Check string representation too
        logging.info(">>> Skipping band selection (Method is None).")
        band_selection_method_log = "None - Used Original"
    elif str(method).upper() == 'SWGMF':
        target_bands = cfg.SWGMF_TARGET_BANDS
        logging.info(f"Target Bands for SWGMF: {target_bands}")
        if target_bands is None or not isinstance(target_bands, int) or target_bands <= 0 or target_bands >= B_original:
             logging.warning(f">>> Skipping SWGMF: Invalid SWGMF_TARGET_BANDS ({target_bands}). Using original bands.")
             band_selection_method_log = f"SWGMF Skipped (Invalid Target) - Used Original"
        else:
            logging.info(">>> Applying SWGMF...")
            band_selection_method_log = f"SWGMF (target={target_bands}, window={cfg.SWGMF_WINDOW_SIZE})"
            try:
                selected_data, selected_indices = bs.apply_swgmf(data_cube, cfg.SWGMF_WINDOW_SIZE, target_bands)
                input_bands = selected_data.shape[-1]
                logging.info(f"Selected band indices (first 10): {selected_indices[:10]}...")
                logging.info(f"Shape after SWGMF: {selected_data.shape}")
                # Add check if selected bands don't match target
                if input_bands != target_bands:
                    logging.warning(f"SWGMF returned {input_bands} bands, but target was {target_bands}. Using the {input_bands} bands found.")
            except Exception as e:
                logging.error(f"Error during SWGMF: {e}", exc_info=True)
                logging.warning("Falling back to original bands.")
                band_selection_method_log = f"SWGMF Failed - Used Original"
                selected_data = data_cube
                input_bands = B_original
                selected_indices = np.arange(B_original)
    elif str(method).upper() == 'E-FDPC':
        logging.info(">>> Applying E-FDPC (Automatic Band Count)...")
        band_selection_method_log = f"E-FDPC (dc_percent={cfg.E_FDPC_DC_PERCENT})"
        try:
            selected_data, selected_indices = bs.apply_efdpc(data_cube, cfg.E_FDPC_DC_PERCENT)
            input_bands = selected_data.shape[-1]
            logging.info(f"Selected band indices (first 10): {selected_indices[:10]}...")
            logging.info(f"Shape after E-FDPC: {selected_data.shape}")
            band_selection_method_log += f" -> Selected {input_bands} bands"
            # Handle case where E-FDPC selects 0 bands
            if input_bands <= 0:
                logging.error("E-FDPC selected 0 bands. Falling back to original bands.")
                band_selection_method_log = f"E-FDPC Failed (0 bands) - Used Original"
                selected_data = data_cube
                input_bands = B_original
                selected_indices = np.arange(B_original)
        except Exception as e:
            logging.error(f"Error during E-FDPC: {e}", exc_info=True)
            logging.warning("Falling back to original bands.")
            band_selection_method_log = f"E-FDPC Failed - Used Original"
            selected_data = data_cube
            input_bands = B_original
            selected_indices = np.arange(B_original)
    else:
        logging.warning(f">>> Unknown BAND_SELECTION_METHOD '{method}'. Using original bands.")
        band_selection_method_log = f"Unknown Method ({method}) - Used Original"

    logging.info(f"Final input bands to model: {input_bands}")
    logging.info(f"Shape of 'selected_data' after band selection step: {selected_data.shape}")
    return selected_data, input_bands, band_selection_method_log

def preprocess_and_split(selected_data, gt_map):
    """Normalizes, pads, splits, and creates patches."""
    logging.info("\n--- Normalizing Data ---")
    logging.info(f"Normalizing data with shape: {selected_data.shape}")
    normalized_data = du.normalize_data(selected_data)
    logging.info("\n--- Padding Data ---")
    padded_data = du.pad_data(normalized_data, cfg.BORDER_SIZE)
    logging.info("\n--- Splitting Data and Creating Patches ---")
    all_labeled_coords, labels_np_array, original_idx_array = spl.get_labeled_coordinates_and_indices(gt_map)
    split_coords_dict = spl.split_data_random_stratified(
        all_labeled_coords, labels_np_array, original_idx_array,
        cfg.TRAIN_RATIO, cfg.VAL_RATIO, cfg.NUM_CLASSES, cfg.RANDOM_SEED
    )
    data_splits = {}
    logging.info("Creating patches for TRAIN set...")
    data_splits['train_patches'], data_splits['train_labels'] = du.create_patches_from_coords(padded_data, split_coords_dict['train_coords'], cfg.PATCH_SIZE)
    logging.info(f"Created {len(data_splits.get('train_patches', []))} training patches.")
    logging.info("Creating patches for VALIDATION set...")
    data_splits['val_patches'], data_splits['val_labels'] = du.create_patches_from_coords(padded_data, split_coords_dict['val_coords'], cfg.PATCH_SIZE)
    logging.info(f"Created {len(data_splits.get('val_patches', []))} validation patches.")
    logging.info("Creating patches for TEST set...")
    data_splits['test_patches'], data_splits['test_labels'] = du.create_patches_from_coords(padded_data, split_coords_dict['test_coords'], cfg.PATCH_SIZE)
    logging.info(f"Created {len(data_splits.get('test_patches', []))} test patches.")
    data_splits['test_coords'] = split_coords_dict['test_coords'] # Keep coords for visualization
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
    train_loader, val_loader, test_loader = loaders
    trained_model, history = None, None
    training_successful = False

    logging.info("\n--- Starting Training ---")
    if train_loader is None:
        logging.error("Cannot train model: Training data loader is not available.")
    else:
        model.to(device)
        # Pass fusion mechanism info implicitly via model instance to engine
        trained_model, history = engine.train_model(
            model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            train_loader=train_loader, val_loader=val_loader, device=device,
            epochs=cfg.EPOCHS, loss_epsilon=cfg.LOSS_EPSILON, use_scheduler=cfg.USE_SCHEDULER
        )
        training_successful = True
        logging.info("Training finished.")

        if cfg.SAVE_BEST_MODEL and val_loader and trained_model:
            best_model_path = os.path.join(run_output_dir, f"{cfg.DATASET_NAME}_best_model.pth")
            # Save model state dict directly
            try:
                 torch.save(trained_model.state_dict(), best_model_path)
                 logging.info(f"Best model weights saved to: {best_model_path}")
            except Exception as e:
                 logging.error(f"Error saving best model weights: {e}")
        elif cfg.SAVE_BEST_MODEL:
            logging.warning("Best model saving enabled, but validation failed or unavailable.")

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
        logging.warning("Model not trained successfully. Skipping evaluation.")
    else:
        logging.warning("Test data loader unavailable. Skipping evaluation.")

    return trained_model, history, training_successful, evaluation_successful, oa, aa, kappa, report, test_preds, test_labels

def save_results_and_visualize(run_output_dir, gt_map, data_splits, history, training_successful, evaluation_successful, oa, aa, kappa, report, test_preds, test_labels, pipeline_duration_seconds, band_selection_method_log, input_bands):
    """Saves results, config, and plots."""
    logging.info("\n--- Saving Run Configuration & Results ---")
    if run_output_dir:
        # Save Run Configuration
        try:
            swgmf_params = None
            efdpc_params = None
            target_bands_config = None
            if "SWGMF" in band_selection_method_log:
                 swgmf_params = {"window_size": cfg.SWGMF_WINDOW_SIZE}
                 target_bands_config = cfg.SWGMF_TARGET_BANDS
            elif "E-FDPC" in band_selection_method_log:
                 efdpc_params = {"dc_percent": cfg.E_FDPC_DC_PERCENT}
                 target_bands_config = "Auto"

            run_config = {
                "run_timestamp": run_timestamp, "dataset_name": cfg.DATASET_NAME,
                "data_file": cfg.DATA_FILE, "gt_file": cfg.GT_FILE, "num_classes": cfg.NUM_CLASSES,
                "band_selection_config": {
                    "method_used": band_selection_method_log,
                    "target_bands_set": target_bands_config,
                    "swgmf_params": swgmf_params,
                    "efdpc_params": efdpc_params,
                 },
                "input_bands_to_model": input_bands,
                "patch_size": cfg.PATCH_SIZE,
                "train_ratio": cfg.TRAIN_RATIO, "val_ratio": cfg.VAL_RATIO,
                "random_seed": cfg.RANDOM_SEED, "batch_size": cfg.BATCH_SIZE,
                "num_workers": cfg.NUM_WORKERS, "pin_memory": cfg.PIN_MEMORY,
                "model_spec_channels": cfg.SPEC_CHANNELS, "model_spatial_channels": cfg.SPATIAL_CHANNELS,
                "intermediate_attention_stages": cfg.INTERMEDIATE_ATTENTION_STAGES, # Log intermediate setting
                "fusion_mechanism": cfg.FUSION_MECHANISM, # Log fusion mechanism
                "cross_attention_heads": cfg.CROSS_ATTENTION_HEADS, # Log head count
                "cross_attention_dropout": cfg.CROSS_ATTENTION_DROPOUT, # Log dropout
                "epochs": cfg.EPOCHS, "learning_rate": cfg.LEARNING_RATE,
                "weight_decay": cfg.WEIGHT_DECAY, "optimizer_type": cfg.OPTIMIZER_TYPE,
                "loss_epsilon": cfg.LOSS_EPSILON, "use_scheduler": cfg.USE_SCHEDULER,
                "scheduler_step_size": cfg.SCHEDULER_STEP_SIZE if cfg.USE_SCHEDULER else None,
                "scheduler_gamma": cfg.SCHEDULER_GAMMA if cfg.USE_SCHEDULER else None,
                "pipeline_duration_seconds": round(pipeline_duration_seconds, 2) if pipeline_duration_seconds is not None else "N/A",
                "training_status": "Success" if training_successful else "Failed/Skipped",
                "evaluation_status": "Success" if evaluation_successful else "Failed/Skipped",
                "evaluation_results": {
                    "OA": f"{oa:.4f}" if evaluation_successful and oa is not None else "N/A",
                    "AA": f"{aa:.4f}" if evaluation_successful and aa is not None else "N/A",
                    "Kappa": f"{kappa:.4f}" if evaluation_successful and kappa is not None else "N/A"
                } if evaluation_successful else {"Status": "Evaluation skipped or failed"}
            }
            config_path = os.path.join(run_output_dir, "run_config.json")
            with open(config_path, 'w') as f: json.dump(run_config, f, indent=4)
            logging.info(f"Run configuration saved to: {config_path}")
        except Exception as e:
            logging.error(f"Error saving run configuration: {e}", exc_info=True)

        # Save Evaluation Results Text File
        if evaluation_successful and report is not None:
            try:
                results_path = os.path.join(run_output_dir, f"{cfg.DATASET_NAME}_test_results.txt")
                with open(results_path, 'w') as f:
                    f.write(f"Band Selection: {band_selection_method_log}\n")
                    f.write(f"Intermediate Attention Stages: {cfg.INTERMEDIATE_ATTENTION_STAGES if cfg.INTERMEDIATE_ATTENTION_STAGES else 'None'}\n")
                    f.write(f"Fusion Mechanism: {cfg.FUSION_MECHANISM}\n") # Add fusion info
                    f.write(f"Input Bands: {input_bands}\n\n")
                    f.write(f"Overall Accuracy (OA): {oa:.4f}\n")
                    f.write(f"Average Accuracy (AA): {aa:.4f}\n")
                    f.write(f"Kappa Coefficient:      {kappa:.4f}\n\n")
                    f.write("Classification Report:\n")
                    f.write(report)
                logging.info(f"Evaluation report saved to {results_path}")
            except Exception as e:
                 logging.error(f"Error saving evaluation text report: {e}", exc_info=True)

        # Save Predictions/Labels Arrays
        if test_preds is not None and test_labels is not None:
            try:
                np.save(os.path.join(run_output_dir, 'test_predictions.npy'), test_preds)
                np.save(os.path.join(run_output_dir, 'test_labels.npy'), test_labels)
                logging.info("Test predictions and labels saved.")
            except Exception as e:
                 logging.error(f"Error saving predictions/labels: {e}", exc_info=True)

        # Plot and Save History
        logging.info("\n--- Plotting Training History ---")
        if history:
            try:
                import matplotlib.pyplot as plt
                plt.switch_backend('Agg') # Use non-interactive backend
                fig, axes = vis.plot_history(history)
                if fig and axes is not None:
                    history_plot_path = os.path.join(run_output_dir, f"{cfg.DATASET_NAME}_training_history.png")
                    fig.savefig(history_plot_path, dpi=300, bbox_inches='tight')
                    logging.info(f"Training history plot saved to {history_plot_path}")
                    plt.close(fig) # Close figure to free memory
                else: logging.warning("Plotting history returned None, skipping save.")
            except Exception as e:
                logging.error(f"Error plotting/saving history: {e}", exc_info=True)
        else: logging.warning("No training history found to plot/save.")

        # Plot and Save Prediction Map
        logging.info("\n--- Visualizing Test Set Predictions ---")
        plot_possible = evaluation_successful and test_preds is not None and \
                        'test_coords' in data_splits and gt_map is not None and \
                        hasattr(cfg, 'CLASS_NAMES') and cfg.CLASS_NAMES is not None and oa is not None
        if plot_possible:
            try:
                import matplotlib.pyplot as plt
                plt.switch_backend('Agg') # Use non-interactive backend
                fig, axes = vis.plot_predictions(
                    gt_map=gt_map, test_predictions=test_preds,
                    test_coords=data_splits['test_coords'], class_names=cfg.CLASS_NAMES,
                    dataset_name=cfg.DATASET_NAME, oa=oa
                )
                if fig and axes is not None:
                    plot_path = os.path.join(run_output_dir, f"{cfg.DATASET_NAME}_classification_map.png")
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    logging.info(f"Classification map saved to {plot_path}")
                    plt.close(fig) # Close figure to free memory
                else: logging.warning("Plotting predictions returned None, skipping save.")
            except Exception as e:
                logging.error(f"Error plotting/saving prediction map: {e}", exc_info=True)
        else: logging.warning("Skipping prediction visualization/saving (missing data or failed evaluation).")
    else: logging.error("Cannot save results: 'run_output_dir' not available.")

# ==================================
#          Main Execution
# ==================================
if __name__ == "__main__":
    logging.info("========== Starting New Run ==========")
    logging.info(f"Script: {__file__ if '__file__' in locals() else 'interactive'}")
    logging.info(f"Full command: {' '.join(sys.argv)}")
    logging.info(f"Timestamp: {run_timestamp}")

    pipeline_start_time = time.time()
    pipeline_duration_seconds = None
    # Initialize variables to ensure they exist even if errors occur early
    trained_model, history = None, None
    training_successful, evaluation_successful = False, False
    oa, aa, kappa, report, test_preds, test_labels = None, None, None, None, None, None
    data_splits = {} # Initialize data_splits
    gt_map = None # Initialize gt_map
    band_selection_method_log = "Not Run"
    input_bands = -1 # Initialize input_bands

    try:
        # --- Setup ---
        device = setup_environment()

        # --- Data Loading ---
        data_cube, gt_map, B_original = load_data()

        # --- Band Selection ---
        selected_data, input_bands, band_selection_method_log = select_bands(data_cube, B_original)

        # Check if band selection resulted in valid bands
        if input_bands <= 0:
            raise ValueError(f"Band selection resulted in {input_bands} bands. Cannot proceed.")

        # --- Preprocessing & Splitting ---
        data_splits = preprocess_and_split(selected_data, gt_map)

        # --- DataLoaders ---
        loaders = prepare_dataloaders(data_splits)
        train_loader, val_loader, test_loader = loaders
        if not train_loader or not test_loader:
             raise ValueError("Train or Test DataLoader could not be created. Cannot proceed.")

        # --- Model ---
        logging.info("\n--- Instantiating the DSSFN model ---")
        logging.info(f"Model input bands: {input_bands}")
        logging.info(f"Using Intermediate Attention Stages: {cfg.INTERMEDIATE_ATTENTION_STAGES if cfg.INTERMEDIATE_ATTENTION_STAGES else 'None'}")
        logging.info(f"Using Final Fusion Mechanism: {cfg.FUSION_MECHANISM}")
        model = DSSFN(
            input_bands=input_bands, num_classes=cfg.NUM_CLASSES, patch_size=cfg.PATCH_SIZE,
            spec_channels=cfg.SPEC_CHANNELS, spatial_channels=cfg.SPATIAL_CHANNELS,
            fusion_mechanism=cfg.FUSION_MECHANISM, # Pass fusion type to model
            cross_attention_heads=cfg.CROSS_ATTENTION_HEADS,
            cross_attention_dropout=cfg.CROSS_ATTENTION_DROPOUT
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
        pipeline_end_time = time.time()
        pipeline_duration_seconds = pipeline_end_time - pipeline_start_time
        logging.info(f"\nTotal pipeline execution time: {pipeline_duration_seconds:.2f} seconds")

        # --- Save & Visualize ---
        save_results_and_visualize(
            run_output_dir, gt_map, data_splits, history, training_successful,
            evaluation_successful, oa, aa, kappa, report, test_preds, test_labels,
            pipeline_duration_seconds, band_selection_method_log, input_bands
        )

        logging.info("========== Run Finished Successfully ==========")

    except Exception as e:
        logging.error(f"An error occurred during the main execution: {e}")
        logging.error(traceback.format_exc()) # Log the full traceback
        logging.info("========== Run Finished with Errors ==========")
        # Attempt to save partial results/logs if possible
        try:
            save_results_and_visualize(
                run_output_dir, gt_map, data_splits, history, training_successful,
                evaluation_successful, oa, aa, kappa, report, test_preds, test_labels,
                pipeline_duration_seconds, band_selection_method_log, input_bands
            )
            logging.info("Attempted to save partial results after error.")
        except Exception as save_e:
            logging.error(f"Could not save partial results after error: {save_e}")

        # Clean up memory on error
        if 'model' in locals(): del model
        if 'optimizer' in locals(): del optimizer
        # ... other variables ...
        if 'device' in locals() and 'cuda' in str(device):
             try:
                 torch.cuda.empty_cache()
             except Exception:
                 pass # Ignore errors during cleanup
        sys.exit(1) # Exit with error code


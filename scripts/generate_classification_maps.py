# scripts/generate_classification_maps.py
# -*- coding: utf-8 -*-
"""
Generates and saves classification maps for specified datasets using
a fixed DSSFN model configuration (E-FDPC band selection, intermediate attention).
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
import matplotlib # Ensure backend is set before pyplot import
matplotlib.use('Agg') # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt


# --- Add src directory to path ---
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
project_root = os.path.abspath(os.path.join(script_dir, '..'))
module_path = os.path.join(project_root, 'src')

if module_path not in sys.path:
    print(f"Adding {module_path} to sys.path")
    sys.path.insert(0, module_path)
else:
    print(f"{module_path} already in sys.path")

# --- Import Custom Modules ---
try:
    import src.config as base_cfg_module
    importlib.reload(base_cfg_module)
    import src.data_utils as du
    import src.band_selection as bs
    import src.sampling as spl
    import src.datasets as ds
    from src.model import DSSFN
    import src.engine as engine
    import src.visualization as vis # For plotting
    print("Custom modules imported successfully.")
except ImportError as e:
    print(f"Error importing modules: {e}\n{traceback.format_exc()}")
    sys.exit(1)

# --- Ignore specific warnings ---
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ==================================
#        Global Script Settings
# ==================================
# Define the fixed configuration for the model runs
FIXED_MODEL_CONFIG = {
    'BAND_SELECTION_METHOD': 'E-FDPC',
    'E_FDPC_DC_PERCENT': 2.5,
    'SWGMF_TARGET_BANDS': None, # Not used for E-FDPC
    'INTERMEDIATE_ATTENTION_STAGES': [1],
    'FUSION_MECHANISM': 'AdaptiveWeight',
    'TRAIN_RATIO': 0.10, # Standard train ratio for consistency
    'VAL_RATIO': 0.10,   # Standard val ratio
}

# Define dataset-specific configurations
DATASET_CONFIGURATIONS = [
    {
        'DATASET_NAME': 'IndianPines',
        'DATA_PATH_RELATIVE': 'ip/',
        'DATA_FILE': 'indianpinearray.npy',
        'GT_FILE': 'IPgt.npy',
        'NUM_CLASSES': 16,
        'EXPECTED_DATA_SHAPE': (145, 145, 200),
        'EXPECTED_GT_SHAPE': (145, 145),
        'CLASS_NAMES': [
            "Background/Untested", "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
            "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
            "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
            "Soybean-clean", "Wheat", "Woods", "Bldg-Grass-Tree-Drives",
            "Stone-Steel-Towers"
        ],
        'DATA_MAT_KEY': None, 'GT_MAT_KEY': None,
    },
    {
        'DATASET_NAME': 'PaviaUniversity',
        'DATA_PATH_RELATIVE': 'pu/',
        'DATA_FILE': 'PaviaU.mat',
        'GT_FILE': 'PaviaU_gt.mat',
        'NUM_CLASSES': 9,
        'EXPECTED_DATA_SHAPE': (610, 340, 103),
        'EXPECTED_GT_SHAPE': (610, 340),
        'CLASS_NAMES': [
            "Background/Untested", "Asphalt", "Meadows", "Gravel", "Trees",
            "Painted metal sheets", "Bare Soil", "Bitumen",
            "Self-Blocking Bricks", "Shadows"
        ],
        'DATA_MAT_KEY': 'paviaU', 'GT_MAT_KEY': 'paviaU_gt',
    },
    {
        'DATASET_NAME': 'Botswana',
        'DATA_PATH_RELATIVE': 'botswana/',
        'DATA_FILE': 'Botswana.mat',
        'GT_FILE': 'Botswana_gt.mat',
        'NUM_CLASSES': 14,
        'EXPECTED_DATA_SHAPE': (1476, 256, 145),
        'EXPECTED_GT_SHAPE': (1476, 256),
        'CLASS_NAMES': [
            "Background/Untested", "Water", "Hippo grass", "Floodplain grasses 1",
            "Floodplain grasses 2", "Reeds 1", "Riparian", "Firescar 2",
            "Island interior", "Acacia woodlands", "Acacia shrublands",
            "Acacia grasslands", "Short mopane", "Mixed mopane", "Exposed soils"
        ],
        'DATA_MAT_KEY': 'Botswana', 'GT_MAT_KEY': 'Botswana_gt',
    }
]

# ==================================
#        Logger Setup
# ==================================
script_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
overall_log_dir = os.path.join(project_root, base_cfg_module.OUTPUT_DIR, "classification_map_generation")
os.makedirs(overall_log_dir, exist_ok=True)
log_file_path = os.path.join(overall_log_dir, f"map_generation_log_{script_run_timestamp}.txt")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

# ==================================
#        Pipeline Functions
# ==================================
def setup_environment(cfg_dict):
    """Sets up device and deterministic settings based on cfg_dict."""
    logging.info(f"--- Setting up Environment (Seed: {cfg_dict['RANDOM_SEED']}) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    torch.manual_seed(cfg_dict['RANDOM_SEED'])
    np.random.seed(cfg_dict['RANDOM_SEED'])
    random.seed(cfg_dict['RANDOM_SEED'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg_dict['RANDOM_SEED'])
        torch.cuda.manual_seed_all(cfg_dict['RANDOM_SEED'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    return device

def load_data(cfg_dict):
    """Loads data based on the provided configuration dictionary."""
    logging.info(f"\n--- Loading Data for {cfg_dict['DATASET_NAME']} ---")
    data_cube, gt_map = du.load_hyperspectral_data(
        cfg_dict['DATA_PATH'], cfg_dict['DATA_FILE'], cfg_dict['GT_FILE'],
        cfg_dict['EXPECTED_DATA_SHAPE'], cfg_dict['EXPECTED_GT_SHAPE'],
        data_mat_key=cfg_dict.get('DATA_MAT_KEY'), gt_mat_key=cfg_dict.get('GT_MAT_KEY')
    )
    H, W, B_original = data_cube.shape
    logging.info(f"Original data: H={H}, W={W}, Bands={B_original}. Classes: {cfg_dict['NUM_CLASSES']}")
    return data_cube, gt_map, B_original

def select_bands(data_cube, cfg_dict):
    """Performs band selection based on cfg_dict."""
    method = cfg_dict['BAND_SELECTION_METHOD']
    logging.info(f"\n--- Band Selection (Method: {method}) ---")
    
    selected_data = data_cube
    input_bands_run = data_cube.shape[-1]
    band_selection_log_detail = "N/A"

    if method == 'E-FDPC':
        dc_percent = cfg_dict['E_FDPC_DC_PERCENT']
        logging.info(f">>> Applying E-FDPC (dc_percent={dc_percent}%)")
        try:
            selected_data, selected_indices = bs.apply_efdpc(data_cube, dc_percent)
            input_bands_run = selected_data.shape[-1]
            band_selection_log_detail = f"E-FDPC (dc={dc_percent}%) -> {input_bands_run} bands"
            if input_bands_run <= 0:
                logging.error("E-FDPC selected 0 bands. Reverting to original.")
                selected_data, input_bands_run = data_cube, data_cube.shape[-1]
                band_selection_log_detail += " - FAILED (0 bands, used original)"
        except Exception as e:
            logging.error(f"Error during E-FDPC: {e}. Using original bands.", exc_info=True)
            selected_data, input_bands_run = data_cube, data_cube.shape[-1]
            band_selection_log_detail = f"E-FDPC (dc={dc_percent}%) - FAILED (used original)"
    elif method == 'SWGMF':
        target_bands = cfg_dict['SWGMF_TARGET_BANDS']
        window_size = cfg_dict['SWGMF_WINDOW_SIZE']
        logging.info(f">>> Applying SWGMF (target={target_bands}, window={window_size})")
        try:
            selected_data, selected_indices = bs.apply_swgmf(data_cube, window_size, target_bands)
            input_bands_run = selected_data.shape[-1]
            band_selection_log_detail = f"SWGMF (target={target_bands}) -> {input_bands_run} bands"
        except Exception as e:
            logging.error(f"Error during SWGMF: {e}. Using original bands.", exc_info=True)
            selected_data, input_bands_run = data_cube, data_cube.shape[-1]
            band_selection_log_detail = f"SWGMF (target={target_bands}) - FAILED (used original)"
    elif method == 'None' or method is None:
        logging.info(">>> No band selection applied. Using original bands.")
        band_selection_log_detail = "None (used original)"
    else:
        logging.warning(f"Unknown band selection method '{method}'. Using original bands.")
        band_selection_log_detail = f"Unknown ({method}, used original)"

    logging.info(f"Band Selection Summary: {band_selection_log_detail}")
    logging.info(f"Shape of data for model: {selected_data.shape}")
    return selected_data, input_bands_run

def preprocess_and_split(selected_data, gt_map, cfg_dict):
    """Normalizes, pads, splits data, and creates patches."""
    logging.info("\n--- Normalizing, Padding, Splitting Data & Creating Patches ---")
    normalized_data = du.normalize_data(selected_data)
    padded_data = du.pad_data(normalized_data, cfg_dict['BORDER_SIZE'])
    
    all_labeled_coords, labels_np_array, original_idx_array = spl.get_labeled_coordinates_and_indices(gt_map)
    if not all_labeled_coords:
        raise ValueError("No labeled pixels found in the ground truth map.")
        
    split_coords_dict = spl.split_data_random_stratified(
        all_labeled_coords, labels_np_array, original_idx_array,
        cfg_dict['TRAIN_RATIO'], cfg_dict['VAL_RATIO'], 
        cfg_dict['NUM_CLASSES'], cfg_dict['RANDOM_SEED']
    )
    
    data_splits = {}
    for split_name in ['train', 'val', 'test']:
        coords = split_coords_dict.get(f'{split_name}_coords', [])
        if coords:
            patches, labels = du.create_patches_from_coords(padded_data, coords, cfg_dict['PATCH_SIZE'])
            data_splits[f'{split_name}_patches'], data_splits[f'{split_name}_labels'] = patches, labels
            logging.info(f"Created {len(patches)} {split_name} patches.")
        else:
            data_splits[f'{split_name}_patches'], data_splits[f'{split_name}_labels'] = np.array([]), np.array([])
            logging.warning(f"No coordinates for {split_name} set.")
            
    data_splits['test_coords'] = split_coords_dict.get('test_coords', [])
    return data_splits

def prepare_dataloaders(data_splits, cfg_dict):
    """Creates DataLoaders."""
    logging.info("\n--- Creating DataLoaders ---")
    loaders = ds.create_dataloaders(
        data_splits, cfg_dict['BATCH_SIZE'], cfg_dict['NUM_WORKERS'], cfg_dict['PIN_MEMORY'])
    if not loaders.get('train') or not loaders.get('test'):
        raise ValueError("Train or Test DataLoader could not be created.")
    return loaders.get('train'), loaders.get('val'), loaders.get('test')

def setup_training(model, cfg_dict):
    """Sets up criterion, optimizer, and scheduler."""
    logging.info("\n--- Setting up training components ---")
    criterion = nn.CrossEntropyLoss()
    optimizer_type = cfg_dict['OPTIMIZER_TYPE'].lower()
    lr = cfg_dict['LEARNING_RATE']
    wd = cfg_dict['WEIGHT_DECAY']

    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else: # Default to SGD
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    logging.info(f"Optimizer: {optimizer_type.upper()} (LR={lr}, WD={wd})")

    scheduler = None
    if cfg_dict['USE_SCHEDULER']:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg_dict['SCHEDULER_STEP_SIZE'], gamma=cfg_dict['SCHEDULER_GAMMA'])
        logging.info(f"Using StepLR scheduler: step={cfg_dict['SCHEDULER_STEP_SIZE']}, gamma={cfg_dict['SCHEDULER_GAMMA']}")
    return criterion, optimizer, scheduler

# ==================================
#          Main Execution
# ==================================
if __name__ == "__main__":
    logging.info("========== Starting Classification Map Generation Script ==========")
    overall_start_time = time.time()

    for dataset_idx, current_dataset_config in enumerate(DATASET_CONFIGURATIONS):
        dataset_name = current_dataset_config['DATASET_NAME']
        logging.info(f"\n\nProcessing Dataset: {dataset_name} ({dataset_idx+1}/{len(DATASET_CONFIGURATIONS)})")
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # --- Create a mutable config for this run ---
        run_cfg = base_cfg_module.__dict__.copy() # Start with base config
        run_cfg.update(current_dataset_config)   # Override with dataset-specifics
        run_cfg.update(FIXED_MODEL_CONFIG)       # Override with fixed model choices
        
        # Update DATA_PATH to be absolute
        run_cfg['DATA_PATH'] = os.path.join(base_cfg_module.DATA_BASE_PATH, run_cfg['DATA_PATH_RELATIVE'])
        
        # Ensure specific config values are set for the model
        base_cfg_module.INTERMEDIATE_ATTENTION_STAGES = run_cfg['INTERMEDIATE_ATTENTION_STAGES']
        base_cfg_module.FUSION_MECHANISM = run_cfg['FUSION_MECHANISM']

        # --- Output Directory for this run's map ---
        dataset_output_dir = os.path.join(project_root, base_cfg_module.OUTPUT_DIR, dataset_name, "classification_maps")
        run_specific_output_dir = os.path.join(dataset_output_dir, f"maprun_{run_timestamp}")
        os.makedirs(run_specific_output_dir, exist_ok=True)
        logging.info(f"Output for {dataset_name} map will be in: {run_specific_output_dir}")

        try:
            # --- Setup Environment ---
            device = setup_environment(run_cfg)

            # --- Load Data ---
            data_cube, gt_map, B_original = load_data(run_cfg)

            # --- Band Selection ---
            selected_data, input_bands_for_model = select_bands(data_cube, run_cfg)
            if input_bands_for_model <= 0:
                logging.error(f"Band selection for {dataset_name} resulted in {input_bands_for_model} bands. Skipping.")
                continue
            run_cfg['INPUT_BANDS_ACTUAL'] = input_bands_for_model # Store actual bands used

            # --- Preprocessing & Splitting ---
            data_splits = preprocess_and_split(selected_data, gt_map, run_cfg)

            # --- DataLoaders ---
            train_loader, val_loader, test_loader = prepare_dataloaders(data_splits, run_cfg)
            if not train_loader or not test_loader:
                logging.error(f"Dataloader creation failed for {dataset_name}. Skipping.")
                continue
            
            # --- Model Instantiation ---
            logging.info(f"\n--- Instantiating DSSFN model for {dataset_name} ---")
            model = DSSFN(
                input_bands=input_bands_for_model,
                num_classes=run_cfg['NUM_CLASSES'],
                patch_size=run_cfg['PATCH_SIZE'],
                spec_channels=run_cfg['SPEC_CHANNELS'],
                spatial_channels=run_cfg['SPATIAL_CHANNELS'],
                fusion_mechanism=run_cfg['FUSION_MECHANISM'],
                cross_attention_heads=run_cfg['CROSS_ATTENTION_HEADS'],
                cross_attention_dropout=run_cfg['CROSS_ATTENTION_DROPOUT']
            ).to(device)
            logging.info(f"Model for {dataset_name} instantiated on {device}.")

            # --- Training Setup ---
            criterion, optimizer, scheduler = setup_training(model, run_cfg)

            # --- Train Model ---
            logging.info(f"\n--- Training model for {dataset_name} ---")
            # Use early stopping settings from run_cfg
            trained_model, history = engine.train_model(
                model, criterion, optimizer, scheduler, train_loader, val_loader, device,
                epochs=run_cfg['EPOCHS'], loss_epsilon=run_cfg['LOSS_EPSILON'],
                use_scheduler=run_cfg['USE_SCHEDULER'],
                save_best_model=run_cfg.get('SAVE_BEST_MODEL', True),
                early_stopping_enabled=run_cfg.get('EARLY_STOPPING_ENABLED', True),
                early_stopping_patience=run_cfg.get('EARLY_STOPPING_PATIENCE', 10),
                early_stopping_metric=run_cfg.get('EARLY_STOPPING_METRIC', 'val_loss'),
                early_stopping_min_delta=run_cfg.get('EARLY_STOPPING_MIN_DELTA', 0.0001)
            )
            logging.info(f"Training finished for {dataset_name}.")

            # --- Evaluate to get predictions and OA ---
            logging.info(f"\n--- Evaluating model for {dataset_name} to get predictions ---")
            oa, aa, kappa, report, test_preds, test_labels = engine.evaluate_model(
                trained_model, test_loader, device, criterion, run_cfg['LOSS_EPSILON']
            )
            logging.info(f"Evaluation for {dataset_name}: OA={oa:.4f}, AA={aa:.4f}, Kappa={kappa:.4f}")
            
            # --- Generate and Save Classification Map ---
            if test_preds is not None and data_splits.get('test_coords'):
                map_file_path = os.path.join(run_specific_output_dir, f"{dataset_name}_classification_map.png")
                logging.info(f"Generating classification map for {dataset_name} at {map_file_path}...")
                
                fig, _ = vis.plot_predictions(
                    gt_map=gt_map,
                    test_predictions=test_preds,
                    test_coords=data_splits['test_coords'],
                    class_names=run_cfg['CLASS_NAMES'],
                    dataset_name=dataset_name,
                    oa=oa 
                )
                if fig:
                    try:
                        fig.savefig(map_file_path, dpi=300, bbox_inches='tight')
                        logging.info(f"Classification map for {dataset_name} saved to {map_file_path}")
                        plt.close(fig) # Close figure to free memory
                    except Exception as e_plot:
                        logging.error(f"Error saving plot for {dataset_name}: {e_plot}", exc_info=True)
                else:
                    logging.warning(f"Plotting function returned no figure for {dataset_name}.")
            else:
                logging.warning(f"Skipping map generation for {dataset_name} due to missing predictions or coordinates.")

            # --- Save run config for this map generation ---
            try:
                run_cfg_to_save = {k: v for k, v in run_cfg.items() if isinstance(v, (int, float, str, list, dict, bool, type(None)))}
                run_cfg_to_save['evaluation_metrics'] = {'OA': oa, 'AA': aa, 'Kappa': kappa}
                run_cfg_to_save['actual_input_bands'] = input_bands_for_model
                config_save_path = os.path.join(run_specific_output_dir, f"{dataset_name}_map_run_config.json")
                with open(config_save_path, 'w') as f:
                    json.dump(run_cfg_to_save, f, indent=4)
                logging.info(f"Run configuration for {dataset_name} map saved to {config_save_path}")
            except Exception as e_json:
                logging.error(f"Error saving run_cfg JSON for {dataset_name}: {e_json}", exc_info=True)


        except Exception as e_dataset:
            logging.error(f"Error processing dataset {dataset_name}: {e_dataset}", exc_info=True)
        finally:
            if 'cuda' in str(device):
                torch.cuda.empty_cache()
            logging.info(f"Finished processing {dataset_name}.")

    overall_end_time = time.time()
    logging.info(f"\n\nTotal script execution time: {(overall_end_time - overall_start_time)/60:.2f} minutes")
    logging.info("========== Classification Map Generation Script Finished ==========")


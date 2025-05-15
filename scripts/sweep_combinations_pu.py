# -*- coding: utf-8 -*-
"""
Standalone Python script for Combination Sweep Experiment with Cross-Validation
for the Pavia University Dataset.

This script systematically tests the DSSFN model performance across a grid of:
- Band Selection Methods (SWGMF(30), E-FDPC(2.5), None)
- Intermediate Attention (After Stage 1 vs. None)
- Final Fusion Mechanism (AdaptiveWeight)

Includes Early Stopping based on validation loss.
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
    importlib.reload(base_cfg_module) # Reload to ensure fresh state
    # --- MODIFICATION: Set Dataset to Pavia University ---
    base_cfg_module.DATASET_NAME = 'PaviaUniversity'
    # Re-run dependent config logic if DATASET_NAME changes affect other base_cfg_module attributes
    if base_cfg_module.DATASET_NAME == 'PaviaUniversity':
        base_cfg_module.DATA_PATH_RELATIVE = 'pu/'
        base_cfg_module.DATA_FILE = 'PaviaU.mat'
        base_cfg_module.GT_FILE = 'PaviaU_gt.mat'
        base_cfg_module.NUM_CLASSES = 9
        base_cfg_module.EXPECTED_DATA_SHAPE = (610, 340, 103)
        base_cfg_module.EXPECTED_GT_SHAPE = (610, 340)
        base_cfg_module.CLASS_NAMES = [
            "Background/Untested", "Asphalt", "Meadows", "Gravel", "Trees",
            "Painted metal sheets", "Bare Soil", "Bitumen",
            "Self-Blocking Bricks", "Shadows"
        ]
        base_cfg_module.DATA_MAT_KEY = 'paviaU'
        base_cfg_module.GT_MAT_KEY = 'paviaU_gt'
    else:
        pass # Handle other datasets if necessary
    base_cfg_module.DATA_PATH = os.path.join(base_cfg_module.DATA_BASE_PATH, base_cfg_module.DATA_PATH_RELATIVE)
    # --- Ensure Early Stopping defaults are loaded from the module ---
    # These will be picked up by config_to_dict
    base_cfg_module.EARLY_STOPPING_ENABLED = getattr(base_cfg_module, 'EARLY_STOPPING_ENABLED', True)
    base_cfg_module.EARLY_STOPPING_PATIENCE = getattr(base_cfg_module, 'EARLY_STOPPING_PATIENCE', 25)
    base_cfg_module.EARLY_STOPPING_METRIC = getattr(base_cfg_module, 'EARLY_STOPPING_METRIC', 'val_loss')
    base_cfg_module.EARLY_STOPPING_MIN_DELTA = getattr(base_cfg_module, 'EARLY_STOPPING_MIN_DELTA', 0.0001)


    print(f"Reloaded base config module and set DATASET_NAME to: {base_cfg_module.DATASET_NAME}")
    print(f"Data path set to: {base_cfg_module.DATA_PATH}")
    if base_cfg_module.EARLY_STOPPING_ENABLED:
        print(f"Early stopping from config: Patience={base_cfg_module.EARLY_STOPPING_PATIENCE}, Metric='{base_cfg_module.EARLY_STOPPING_METRIC}'")


    import src.data_utils as du
    import src.band_selection as bs
    import src.sampling as spl
    import src.datasets as ds
    from src.model import DSSFN
    import src.engine as engine
    print("Custom modules imported successfully using 'src.' prefix.")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Please ensure the 'src' directory exists at: {module_path}")
    print(f"Current sys.path: {sys.path}")
    raise

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
        if not key.startswith("__") and \
           not isinstance(getattr(cfg_module, key), (types.ModuleType, types.FunctionType, types.BuiltinFunctionType)):
            value = getattr(cfg_module, key)
            # Ensure all relevant types are copied
            if isinstance(value, (int, float, str, list, dict, bool, type(None), tuple, np.ndarray)):
                 cfg_dict[key] = copy.deepcopy(value)
    return cfg_dict

# ==================================
#        Sweep Parameters
# ==================================
band_selection_options = [
    {'method': 'SWGMF', 'param': 30},
    {'method': 'E-FDPC', 'param': 2.5},
    {'method': 'None', 'param': None}
]
intermediate_attention_options = [
    [],
    [1]
]
fusion_mechanism_options = [
    'AdaptiveWeight'
]

parameter_combinations = list(itertools.product(
    band_selection_options,
    intermediate_attention_options,
    fusion_mechanism_options
))
num_combinations = len(parameter_combinations)


K_FOLDS = 10
BASE_RANDOM_SEED = base_cfg_module.RANDOM_SEED
FOLD_TRAIN_RATIO = 0.10
FOLD_VAL_RATIO = 0.10

# ==================================
#        Logger Setup
# ==================================
sweep_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sweep_output_dir = os.path.join(project_root, base_cfg_module.OUTPUT_DIR, base_cfg_module.DATASET_NAME, f"sweep_combinations_pu_{sweep_timestamp}")
os.makedirs(sweep_output_dir, exist_ok=True)
log_file_path = os.path.join(sweep_output_dir, "sweep_combinations_pu_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info(f"========== Starting Pavia University Combination Sweep Run (K-Fold CV) ==========")
logging.info(f"Script: {__file__ if '__file__' in locals() else 'interactive'}")
logging.info(f"Full command: {' '.join(sys.argv)}")
logging.info(f"Sweep Timestamp: {sweep_timestamp}")
logging.info(f"Parameter Combinations per fold: {num_combinations}")
logging.info(f"K-Folds: {K_FOLDS}")
logging.info(f"Base Random Seed: {BASE_RANDOM_SEED}")
logging.info(f"Dataset: {base_cfg_module.DATASET_NAME}")
if base_cfg_module.EARLY_STOPPING_ENABLED:
    logging.info(f"Early Stopping is ENABLED globally: Patience={base_cfg_module.EARLY_STOPPING_PATIENCE}, Metric='{base_cfg_module.EARLY_STOPPING_METRIC}', MinDelta={base_cfg_module.EARLY_STOPPING_MIN_DELTA}")
else:
    logging.info("Early Stopping is DISABLED globally.")


# ==================================
# Pipeline Functions (Adapted to use config dictionary)
# ==================================
def setup_environment(seed):
    logging.info(f"--- Setting up Environment (Seed: {seed}) ---")
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
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    return device

def load_data(cfg_dict):
    logging.info("\n--- Loading Data ---")
    data_cube, gt_map = du.load_hyperspectral_data(
        cfg_dict['DATA_PATH'], cfg_dict['DATA_FILE'], cfg_dict['GT_FILE'],
        cfg_dict['EXPECTED_DATA_SHAPE'], cfg_dict['EXPECTED_GT_SHAPE'],
        data_mat_key=cfg_dict.get('DATA_MAT_KEY'), gt_mat_key=cfg_dict.get('GT_MAT_KEY')
    )
    H, W, B_original = data_cube.shape
    logging.info(f"Original data: H={H}, W={W}, Bands={B_original}. Classes: {cfg_dict['NUM_CLASSES']}")
    return data_cube, gt_map, B_original

def select_bands_for_run(data_cube, B_original, band_option, cfg_dict):
    method = band_option['method']
    param = band_option['param']
    logging.info(f"\n--- Band Selection (Method: {method}, Param: {param}) ---")
    selected_data, input_bands_run, band_selection_method_run = data_cube, B_original, f"{method}"
    if method == 'SWGMF':
        target_bands = param
        band_selection_method_run += f"(target={target_bands})"
        if target_bands and 0 < target_bands < B_original:
            selected_data, _ = bs.apply_swgmf(data_cube, cfg_dict['SWGMF_WINDOW_SIZE'], target_bands)
            input_bands_run = selected_data.shape[-1]
        else: logging.warning(f"Invalid SWGMF target_bands: {target_bands}. Using original.")
    elif method == 'E-FDPC':
        dc_percent = param
        band_selection_method_run += f"(dc%={dc_percent})"
        selected_data, _ = bs.apply_efdpc(data_cube, dc_percent)
        input_bands_run = selected_data.shape[-1]
        if input_bands_run <=0:
            logging.error("E-FDPC selected 0 bands. Reverting to original.")
            selected_data, input_bands_run = data_cube, B_original
    logging.info(f"Band Selection: {band_selection_method_run}. Input bands for model: {input_bands_run}")
    return selected_data, input_bands_run, band_selection_method_run

def preprocess_and_split(selected_data, gt_map, split_coords_dict, cfg_dict):
    normalized_data = du.normalize_data(selected_data)
    padded_data = du.pad_data(normalized_data, cfg_dict['BORDER_SIZE'])
    data_splits = {}
    for split_name in ['train', 'val', 'test']:
        coords = split_coords_dict.get(f'{split_name}_coords', [])
        if coords:
            data_splits[f'{split_name}_patches'], data_splits[f'{split_name}_labels'] = \
                du.create_patches_from_coords(padded_data, coords, cfg_dict['PATCH_SIZE'])
        else:
            data_splits[f'{split_name}_patches'], data_splits[f'{split_name}_labels'] = np.array([]), np.array([])
    data_splits['test_coords'] = split_coords_dict.get('test_coords', [])
    return data_splits

def prepare_dataloaders(data_splits, cfg_dict):
    loaders = ds.create_dataloaders(
        data_splits, cfg_dict['BATCH_SIZE'], cfg_dict['NUM_WORKERS'], cfg_dict['PIN_MEMORY'])
    return loaders.get('train'), loaders.get('val'), loaders.get('test')

def setup_training(model, cfg_dict):
    criterion = nn.CrossEntropyLoss()
    opt_type = cfg_dict['OPTIMIZER_TYPE'].lower()
    if opt_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg_dict['LEARNING_RATE'], weight_decay=cfg_dict['WEIGHT_DECAY'])
    elif opt_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg_dict['LEARNING_RATE'], weight_decay=cfg_dict['WEIGHT_DECAY'])
    else: # Default or SGD
        optimizer = optim.SGD(model.parameters(), lr=cfg_dict['LEARNING_RATE'], momentum=0.9, weight_decay=cfg_dict['WEIGHT_DECAY'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg_dict['SCHEDULER_STEP_SIZE'], gamma=cfg_dict['SCHEDULER_GAMMA']) if cfg_dict['USE_SCHEDULER'] else None
    return criterion, optimizer, scheduler

def train_evaluate_model(model, criterion, optimizer, scheduler, loaders, device, cfg_dict):
    train_loader, val_loader, test_loader = loaders
    trained_model, history = None, None
    training_successful = False
    logging.info("\n--- Starting Training ---")
    if cfg_dict.get('EARLY_STOPPING_ENABLED', False) and val_loader:
        logging.info(f"Early stopping is ACTIVE for this run: Patience={cfg_dict['EARLY_STOPPING_PATIENCE']}, Metric='{cfg_dict['EARLY_STOPPING_METRIC']}'")
    elif cfg_dict.get('EARLY_STOPPING_ENABLED', False) and not val_loader:
        logging.warning("Early stopping configured but no validation loader. It will be disabled for this run.")


    if train_loader and len(train_loader.dataset) > 0:
        model.to(device)
        trained_model, history = engine.train_model(
            model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            train_loader=train_loader, val_loader=val_loader, device=device,
            epochs=cfg_dict['EPOCHS'], loss_epsilon=cfg_dict['LOSS_EPSILON'],
            use_scheduler=cfg_dict['USE_SCHEDULER'],
            save_best_model=cfg_dict.get('SAVE_BEST_MODEL', True), # Pass from config
            early_stopping_enabled=cfg_dict.get('EARLY_STOPPING_ENABLED', False), # Pass from config
            early_stopping_patience=cfg_dict.get('EARLY_STOPPING_PATIENCE', 10), # Pass from config
            early_stopping_metric=cfg_dict.get('EARLY_STOPPING_METRIC', 'val_loss'), # Pass from config
            early_stopping_min_delta=cfg_dict.get('EARLY_STOPPING_MIN_DELTA', 0.0001) # Pass from config
        )
        training_successful = True
    else:
        logging.error("Training loader empty or None. Skipping training.")

    logging.info("\n--- Evaluating Model on Test Set ---")
    oa, aa, kappa, report = None, None, None, None
    evaluation_successful = False
    if trained_model and test_loader and len(test_loader.dataset) > 0:
        trained_model.to(device) # Ensure model is on correct device for eval
        oa, aa, kappa, report, _, _ = engine.evaluate_model(
            model=trained_model, test_loader=test_loader, device=device,
            criterion=criterion, loss_epsilon=cfg_dict['LOSS_EPSILON']
        )
        evaluation_successful = True
    else:
        logging.warning("Model not trained or test loader empty. Skipping evaluation.")
    return trained_model, history, training_successful, evaluation_successful, oa, aa, kappa, report

def format_results_table(df_agg):
    float_cols = ['OA_mean', 'OA_std', 'AA_mean', 'AA_std', 'Kappa_mean', 'Kappa_std', 'Time_mean', 'Time_std']
    for col in float_cols:
        if col in df_agg.columns:
            df_agg[col] = df_agg[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    display_cols = [
        'Band Selection', 'Band Param', 'Int. Attention', 'Fusion', 'Successful Folds',
        'OA_mean', 'OA_std', 'AA_mean', 'AA_std', 'Kappa_mean', 'Kappa_std',
        'Time_mean', 'Time_std'
    ]
    display_cols = [col for col in display_cols if col in df_agg.columns]
    return df_agg[display_cols].to_string(index=False)

# ==================================
# Main Execution
# ==================================
if __name__ == "__main__":
    overall_start_time = time.time()
    all_run_results = []

    try:
        base_cfg_dict = config_to_dict(base_cfg_module)
        logging.info(f"Base config includes EARLY_STOPPING_ENABLED: {base_cfg_dict.get('EARLY_STOPPING_ENABLED')}")

        data_cube, gt_map, B_original = load_data(base_cfg_dict)
        all_labeled_coords, labels_np_array, original_idx_array = \
            spl.get_labeled_coordinates_and_indices(gt_map)
        if not all_labeled_coords: raise ValueError("No labeled pixels found.")

        for fold in range(K_FOLDS):
            fold_start_time = time.time()
            current_seed = BASE_RANDOM_SEED + fold
            logging.info(f"\n\n======= Starting Fold {fold+1}/{K_FOLDS} (Seed: {current_seed}) =======")
            device = setup_environment(current_seed)
            split_coords_dict = spl.split_data_random_stratified(
                all_labeled_coords, labels_np_array, original_idx_array,
                FOLD_TRAIN_RATIO, FOLD_VAL_RATIO, base_cfg_dict['NUM_CLASSES'], current_seed
            )

            for combo_idx, (band_option, intermediate_attention, fusion_mechanism) in enumerate(parameter_combinations):
                run_start_time = time.time()
                logging.info(f"\n--- Fold {fold+1}, Combo {combo_idx+1}/{num_combinations}: BS={band_option}, IA={intermediate_attention}, Fusion={fusion_mechanism} ---")

                run_cfg_dict = base_cfg_dict.copy()
                run_cfg_dict['RANDOM_SEED'] = current_seed
                run_cfg_dict['INTERMEDIATE_ATTENTION_STAGES'] = intermediate_attention
                run_cfg_dict['FUSION_MECHANISM'] = fusion_mechanism
                # Early stopping parameters are already in run_cfg_dict from base_cfg_dict

                run_status = "Success"
                oa, aa, kappa = None, None, None
                input_bands_run = B_original

                try:
                    selected_data, input_bands_run, _ = select_bands_for_run(
                        data_cube, B_original, band_option, run_cfg_dict
                    )
                    if input_bands_run <= 0:
                        run_status = "Skipped (Invalid Bands)"
                        all_run_results.append({
                            'Fold': fold + 1, 'Seed': current_seed, 'Band Selection': band_option['method'],
                            'Band Param': band_option['param'], 'Int. Attention Stages': str(intermediate_attention),
                            'Fusion Mechanism': fusion_mechanism, 'Input Bands': input_bands_run,
                            'OA': None, 'AA': None, 'Kappa': None, 'Run Time (s)': time.time() - run_start_time, 'Status': run_status
                        })
                        continue

                    data_splits = preprocess_and_split(selected_data, gt_map, split_coords_dict, run_cfg_dict)
                    loaders = prepare_dataloaders(data_splits, run_cfg_dict)
                    train_loader, _, test_loader = loaders # val_loader is also returned but used directly in train_evaluate

                    if not train_loader or len(train_loader.dataset) == 0 or not test_loader or len(test_loader.dataset) == 0:
                        run_status = "Skipped (DataLoader Error)"
                    else:
                        model = DSSFN(
                            input_bands=input_bands_run, num_classes=run_cfg_dict['NUM_CLASSES'],
                            patch_size=run_cfg_dict['PATCH_SIZE'], spec_channels=run_cfg_dict['SPEC_CHANNELS'],
                            spatial_channels=run_cfg_dict['SPATIAL_CHANNELS'], fusion_mechanism=run_cfg_dict['FUSION_MECHANISM'],
                            cross_attention_heads=run_cfg_dict['CROSS_ATTENTION_HEADS'],
                            cross_attention_dropout=run_cfg_dict['CROSS_ATTENTION_DROPOUT']
                        ).to(device)
                        criterion, optimizer, scheduler = setup_training(model, run_cfg_dict)
                        _, _, training_successful, evaluation_successful, oa, aa, kappa, _ = \
                            train_evaluate_model(model, criterion, optimizer, scheduler, loaders, device, run_cfg_dict)
                        if not training_successful or not evaluation_successful:
                            run_status = "Failed (Training/Evaluation)"
                except Exception as e_run:
                    logging.error(f"Error in run: {e_run}", exc_info=True)
                    run_status = f"Error ({type(e_run).__name__})"
                finally:
                    run_duration_seconds = time.time() - run_start_time
                    all_run_results.append({
                        'Fold': fold + 1, 'Seed': current_seed, 'Band Selection': band_option['method'],
                        'Band Param': band_option['param'], 'Int. Attention Stages': str(intermediate_attention),
                        'Fusion Mechanism': fusion_mechanism, 'Input Bands': input_bands_run,
                        'OA': oa, 'AA': aa, 'Kappa': kappa, 'Run Time (s)': run_duration_seconds, 'Status': run_status
                    })
                    logging.info(f"Run Time: {run_duration_seconds:.2f}s. Result: OA={oa if oa else 'N/A'}, Status: {run_status}")
                    if 'cuda' in str(device): torch.cuda.empty_cache()
            logging.info(f"======= Fold {fold+1} finished in {time.time() - fold_start_time:.2f} sec =======")

        logging.info("\n\n========== All Folds and Combinations Finished ==========")
        if all_run_results:
            df_detailed = pd.DataFrame(all_run_results)
            df_detailed['Band Param'] = df_detailed['Band Param'].fillna('N/A')
            df_detailed.to_csv(os.path.join(sweep_output_dir, f"sweep_combinations_pu_detailed_results_{sweep_timestamp}.csv"), index=False)
            logging.info(f"Detailed results saved.")

            df_valid_eval = df_detailed.dropna(subset=['OA', 'AA', 'Kappa'])
            if not df_valid_eval.empty:
                agg_funcs = {'OA': ['mean', 'std'], 'AA': ['mean', 'std'], 'Kappa': ['mean', 'std'],
                             'Run Time (s)': ['mean', 'std'], 'Fold': ['count']}
                df_aggregated = df_valid_eval.groupby(['Band Selection', 'Band Param', 'Int. Attention Stages', 'Fusion Mechanism']).agg(agg_funcs)
                df_aggregated.columns = ['_'.join(col).strip() for col in df_aggregated.columns.values]
                df_aggregated = df_aggregated.rename(columns={'Fold_count': 'Successful Folds'}).reset_index()
                df_aggregated = df_aggregated.rename(columns={'Int. Attention Stages': 'Int. Attention', 'Fusion Mechanism': 'Fusion',
                                                              'Run Time (s)_mean': 'Time_mean', 'Run Time (s)_std': 'Time_std'})
                logging.info("Aggregated Results:\n" + df_aggregated.to_string())
                df_aggregated.to_csv(os.path.join(sweep_output_dir, f"sweep_combinations_pu_aggregated_results_{sweep_timestamp}.csv"), index=False)
                with open(os.path.join(sweep_output_dir, f"sweep_combinations_pu_aggregated_results_{sweep_timestamp}.txt"), 'w') as f:
                    f.write(f"Aggregated Results (Pavia University) - Timestamp: {sweep_timestamp}\n{format_results_table(df_aggregated.copy())}")
                logging.info(f"Aggregated results saved.")
            else: logging.warning("No runs completed successfully for aggregation.")
        else: logging.warning("No results collected.")

    except Exception as e:
        logging.error(f"Critical error in sweep: {e}", exc_info=True)
        if all_run_results:
            pd.DataFrame(all_run_results).to_csv(os.path.join(sweep_output_dir, f"sweep_combinations_pu_DETAILED_ERROR_{sweep_timestamp}.csv"), index=False)
        sys.exit(1)
    finally:
        logging.info(f"\nTotal sweep time: {(time.time() - overall_start_time)/3600:.2f} hours")
        logging.info("========== Pavia University Combination Sweep Run Finished ==========")


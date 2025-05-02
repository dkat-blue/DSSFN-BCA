# src/config.py
# Configuration settings for the DSSFN model and experiments
# Uses absolute paths derived from project root.

import os

# --- Calculate Project Root ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Project Root calculated in config.py: {PROJECT_ROOT}") # For verification

# --- Base Paths (Absolute) ---
DATA_BASE_PATH = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results') # Base directory for all outputs

# --- Dataset Configuration ---
DATASET_NAME = 'IndianPines' # 'IndianPines', 'PaviaUniversity', 'Salinas', 'KennedySpaceCenter'

# --- Data Paths (Specific to DATASET_NAME, now relative to DATA_BASE_PATH) ---
if DATASET_NAME == 'IndianPines':
    DATA_PATH_RELATIVE = 'ip/'
    DATA_FILE = 'indianpinearray.npy'
    GT_FILE = 'IPgt.npy'
    NUM_CLASSES = 16
    EXPECTED_DATA_SHAPE = (145, 145, 200)
    EXPECTED_GT_SHAPE = (145, 145)
elif DATASET_NAME == 'PaviaUniversity':
    DATA_PATH_RELATIVE = 'pu/'
    DATA_FILE = 'PaviaU.npy'
    GT_FILE = 'PaviaU_gt.npy'
    NUM_CLASSES = 9
    EXPECTED_DATA_SHAPE = (610, 340, 103)
    EXPECTED_GT_SHAPE = (610, 340)
elif DATASET_NAME == 'Salinas':
    DATA_PATH_RELATIVE = 'sa/'
    DATA_FILE = 'Salinas_corrected.npy'
    GT_FILE = 'Salinas_gt.npy'
    NUM_CLASSES = 16
    EXPECTED_DATA_SHAPE = (512, 217, 204)
    EXPECTED_GT_SHAPE = (512, 217)
elif DATASET_NAME == 'KennedySpaceCenter':
    DATA_PATH_RELATIVE = 'ksc/'
    DATA_FILE = 'KSC.npy'
    GT_FILE = 'KSC_gt.npy'
    NUM_CLASSES = 13
    EXPECTED_DATA_SHAPE = (512, 614, 176)
    EXPECTED_GT_SHAPE = (512, 614)
else:
    raise ValueError(f"Unsupported DATASET_NAME: {DATASET_NAME}")

# Construct the absolute DATA_PATH
DATA_PATH = os.path.join(DATA_BASE_PATH, DATA_PATH_RELATIVE)

# --- Band Selection Parameters ---
BAND_SELECTION_METHOD = 'None' # 'SWGMF', 'E-FDPC', or 'None'
SWGMF_TARGET_BANDS = 30
SWGMF_WINDOW_SIZE = 5
E_FDPC_DC_PERCENT = 2.0 # Not used if method is SWGMF

# --- Patch Extraction Parameters ---
PATCH_SIZE = 15
BORDER_SIZE = PATCH_SIZE // 2

# --- Data Splitting Parameters ---
TRAIN_RATIO = 0.10
VAL_RATIO = 0.10
RANDOM_SEED = 42

# --- DataLoader Parameters ---
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True

# --- Model Architecture Parameters ---
SPEC_CHANNELS = [64, 128, 256]
SPATIAL_CHANNELS = [64, 128, 256] # Ensure channel dims match for enabled INTERMEDIATE_ATTENTION_STAGES

# Control intermediate attention stages
# List containing the stages AFTER which intermediate attention should be applied.
# Example: [1] -> Apply after stage 1 only
# Example: [1, 2] -> Apply after stage 1 and stage 2
# Example: [] -> Disable intermediate attention
INTERMEDIATE_ATTENTION_STAGES = [] # Set to [] to disable intermediate attention

# --- Fusion Mechanism ---
FUSION_MECHANISM = 'AdaptiveWeight' # 'AdaptiveWeight' or 'CrossAttention' for FINAL fusion
# Parameters for CrossAttention (only used if FUSION_MECHANISM is 'CrossAttention')
CROSS_ATTENTION_HEADS = 8 # Number of attention heads (used for intermediate and final if enabled)
CROSS_ATTENTION_DROPOUT = 0.1 # Dropout rate in the attention module output

# --- Training Hyperparameters ---
EPOCHS = 100
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
OPTIMIZER_TYPE = 'AdamW'
LOSS_EPSILON = 1e-8 # Used only for AdaptiveWeight fusion

# --- Learning Rate Scheduler Parameters ---
USE_SCHEDULER = True
SCHEDULER_STEP_SIZE = 30
SCHEDULER_GAMMA = 0.1

# --- Output/Saving Configuration ---
SAVE_BEST_MODEL = True

# --- Visualization ---
if DATASET_NAME == 'IndianPines':
    CLASS_NAMES = [
        "Background / Not Tested", "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
        "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
        "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat",
        "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"
    ]
elif DATASET_NAME == 'PaviaUniversity':
     CLASS_NAMES = [
         "Background", "Asphalt", "Meadows", "Gravel", "Trees",
         "Painted metal sheets", "Bare Soil", "Bitumen",
         "Self-Blocking Bricks", "Shadows"
     ]
elif DATASET_NAME == 'Salinas':
     CLASS_NAMES = [
        "Background", "Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow",
        "Fallow_rough_plow", "Fallow_smooth", "Stubble", "Celery", "Grapes_untrained",
        "Soil_vinyard_develop", "Corn_senesced_green_weeds", "Lettuce_romaine_4wk",
        "Lettuce_romaine_5wk", "Lettuce_romaine_6wk", "Lettuce_romaine_7wk",
        "Vinyard_untrained", "Vinyard_vertical_trellis"
     ]
elif DATASET_NAME == 'KennedySpaceCenter':
     CLASS_NAMES = [
        "Background", "Scrub", "Willow swamp", "CP hammock", "Slash pine",
        "Oak/Broadleaf", "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
        "Cattail marsh", "Salt marsh", "Mud flats", "Water", "Upland Hardwood"
     ]


if 'CLASS_NAMES' in locals() and len(CLASS_NAMES) != NUM_CLASSES + 1:
     print(f"Warning: CLASS_NAMES length mismatch for {DATASET_NAME}")

# --- Verification Prints ---
print(f"Selected Band Selection Method: {BAND_SELECTION_METHOD}")
if BAND_SELECTION_METHOD.upper() == 'SWGMF':
    print(f"SWGMF Target Bands: {SWGMF_TARGET_BANDS}")
elif BAND_SELECTION_METHOD.upper() == 'E-FDPC':
    print(f"E-FDPC auto bands (dc_percent={E_FDPC_DC_PERCENT}).")
else:
    print("No band selection.")

print(f"Intermediate Attention Stages Enabled: {INTERMEDIATE_ATTENTION_STAGES if INTERMEDIATE_ATTENTION_STAGES else 'None'}")
# Check channel matching for enabled stages
if 1 in INTERMEDIATE_ATTENTION_STAGES and SPEC_CHANNELS[0] != SPATIAL_CHANNELS[0]:
    print(f"WARNING: Intermediate attention enabled after Stage 1, but SPEC_CHANNELS[0] ({SPEC_CHANNELS[0]}) != SPATIAL_CHANNELS[0] ({SPATIAL_CHANNELS[0]}).")
if 2 in INTERMEDIATE_ATTENTION_STAGES and SPEC_CHANNELS[1] != SPATIAL_CHANNELS[1]:
    print(f"WARNING: Intermediate attention enabled after Stage 2, but SPEC_CHANNELS[1] ({SPEC_CHANNELS[1]}) != SPATIAL_CHANNELS[1] ({SPATIAL_CHANNELS[1]}).")

print(f"Selected Final Fusion Mechanism: {FUSION_MECHANISM}")
if FUSION_MECHANISM.upper() == 'CROSSATTENTION':
    print(f"Cross-Attention Heads: {CROSS_ATTENTION_HEADS}")
    # Check divisibility for final fusion
    final_dim = SPATIAL_CHANNELS[-1]
    if SPEC_CHANNELS[-1] != final_dim:
         print(f"WARNING: Final CrossAttention selected, but SPEC_CHANNELS[-1] ({SPEC_CHANNELS[-1]}) != SPATIAL_CHANNELS[-1] ({final_dim}).")
    if final_dim % CROSS_ATTENTION_HEADS != 0:
        print(f"WARNING: Final channel dimension ({final_dim}) is not divisible by CROSS_ATTENTION_HEADS ({CROSS_ATTENTION_HEADS}).")

print(f"Absolute Data Path set in config.py: {DATA_PATH}")
print(f"Absolute Base Output Dir set in config.py: {OUTPUT_DIR}")


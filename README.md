# Modified DSSFN Architecture for Hyperspectral Image Classification

This project implements a modified version of the DSSFN (Dual-Stream Self-Attention Fusion Network) architecture, based on the paper ["DSSFN: A Dual-Stream Self-Attention Fusion Network for Effective Hyperspectral Image Classification" (Yang et al., Remote Sensing 2023)](https://www.mdpi.com/2072-4292/15/15/3701), for the classification of Hyperspectral Images (HSI).

## Overview

Hyperspectral images contain rich spectral information, enabling detailed analysis of materials on the Earth's surface. This project utilizes deep learning to classify HSI pixels. The DSSFN architecture processes both spectral and spatial information using two parallel streams.

**Key Components of the Original Architecture:**

* **Dual-Stream:** One stream processes 1D spectral vectors, the other processes 2D spatial patches.
* **Pyramidal Residual Blocks:** Used in both streams to extract features at different levels with residual connections.
* **Self-Attention:** Integrated within the residual blocks to identify the importance of different parts of the spectral/spatial input.

**Modifications in this Implementation:**

* **Optional Intermediate Cross-Attention:** Allows adding `MultiHeadCrossAttention` layers between streams after intermediate stages (Stage 1 and/or Stage 2) for early information exchange.
* **Configurable Fusion Mechanism:** Provides a choice between:
    * `AdaptiveWeight`: Weights the outputs (logits) of each stream based on their current loss.
    * `CrossAttention`: Uses `MultiHeadCrossAttention` to fuse the final features from both streams before a single classification layer.
* **Learned Positional Embeddings:** Used in conjunction with cross-attention (intermediate and final).
* **Flexible Band Selection:** Supports `SWGMF`, `E-FDPC` methods, or using all original bands.

## Project Structure
```
📦 
.gitignore
README.md
data
│  ├─ botswana
│  ├─ ip
│  └─ pu
├─ requirements.txt
├─ results
│  ├─ Botswana
│  ├─ IndianPines
│  └─ PaviaUniversity
├─ scripts
└─ src
```
## Requirements

* Python 3.10+
* PyTorch
* NumPy
* Scikit-learn
* Matplotlib
* Pandas (for sweep result tables)

**Installation:**

1.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
2.  **Install Dependencies:**
    Install the required packages using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure you have the correct base Python and pip versions. The `requirements.txt` file specifies exact versions used during development. You might need to install the correct PyTorch version separately based on your CUDA setup if the one in `requirements.txt` is not compatible: see https://pytorch.org/)*

## Usage

### 1. Data Preparation

* Place your Hyperspectral Image (HSI) datasets into the `data/` directory. These are organized by subdirectories (e.g., [=`data/ip/`](https://www.kaggle.com/datasets/abhijeetgo/indian-pines-hyperspectral-dataset), [`data/pu/`](https://www.kaggle.com/datasets/syamkakarla/pavia-university-hsi), [`data/botswana/`](https://www.kaggle.com/datasets/mingliu123/botswana)).
* Supported data formats are `.npy` (NumPy arrays) and `.mat` (MATLAB files).
* For `.mat` files, ensure you know the variable names (keys) within the file that correspond to the data cube and the ground truth map. These keys will be needed for configuration.

### 2. Configuration (`src/config.py`)

The `src/config.py` file serves as the **base configuration** for various parameters.

* **For `scripts.main` (single runs):**
    * You **must** modify `src/config.py` to define the `DATASET_NAME` you wish to process.
    * Update dataset-specific parameters within `src/config.py` for the chosen `DATASET_NAME`, including:
        * `DATA_PATH_RELATIVE`: Relative path from `data/` to the dataset's folder.
        * `DATA_FILE`, `GT_FILE`: Filenames for the data and ground truth.
        * `NUM_CLASSES`, `EXPECTED_DATA_SHAPE`, `EXPECTED_GT_SHAPE`, `CLASS_NAMES`.
        * `DATA_MAT_KEY`, `GT_MAT_KEY` (if using `.mat` files).
    * Configure other parameters such as:
        * **Band Selection:** `BAND_SELECTION_METHOD` (`'SWGMF'`, `'E-FDPC'`, or `'None'`), `SWGMF_TARGET_BANDS`, `E_FDPC_DC_PERCENT`.
        * **Patch Extraction:** `PATCH_SIZE`.
        * **Data Splitting:** `TRAIN_RATIO`, `VAL_RATIO`, `RANDOM_SEED`.
        * **Model Architecture:** `SPEC_CHANNELS`, `SPATIAL_CHANNELS`, `INTERMEDIATE_ATTENTION_STAGES`, `FUSION_MECHANISM`, `CROSS_ATTENTION_HEADS`, `CROSS_ATTENTION_DROPOUT`.
        * **Training Hyperparameters:** `EPOCHS`, `LEARNING_RATE`, `WEIGHT_DECAY`, `OPTIMIZER_TYPE`, `USE_SCHEDULER`, etc.
        * **Early Stopping:** `EARLY_STOPPING_ENABLED`, `EARLY_STOPPING_PATIENCE`, etc.
* **For Sweep Scripts (`scripts.sweep_bands`, `scripts.sweep_bands_efdpc`):**
    * These scripts use the `DATASET_NAME` (and its associated parameters like paths, num_classes) currently active in `src/config.py`.
    * They iterate through specific parameters (e.g., number of bands for `sweep_bands` or `dc_percent` for `sweep_bands_efdpc`) which are defined *within the sweep scripts themselves*. Other general settings (like epochs, learning rate, model architecture) are taken from `src/config.py`.
* **For `scripts.sweep_master` and `scripts.generate_classification_maps`:**
    * These scripts manage dataset configurations (like paths, number of classes, etc.) *internally*.
    * `sweep_master` uses dataset parameters defined in its `DATASET_SPECIFIC_CONFIGS` dictionary.
    * `generate_classification_maps` uses dataset parameters from its `DATASET_CONFIGURATIONS` list.
    * General hyperparameters (e.g., epochs, learning rates, base model architecture like `SPEC_CHANNELS`) are still typically drawn from `src/config.py` as a starting point, but can be overridden by the logic within these more complex scripts.

### 3. Running Experiments

Ensure your virtual environment is activated. **Run these commands from the project's root directory (the directory containing `scripts/`, `src/`, `data/` etc.).**

* **Single Run (`scripts.main`):**
    * **Purpose:** Execute a single experiment based on the current settings in `src/config.py`.
    * **Configuration:** Modify `src/config.py` as described above for the dataset and parameters you want to test.
    * **Command:**
        ```bash
        python -m scripts.main
        ```
    * **Output:** Results (logs, configuration JSON, evaluation metrics, plots, saved model if enabled) are saved in `results/[DATASET_NAME]/run_[TIMESTAMP]/`.

* **SWGMF Band Sweep (`scripts.sweep_bands`):**
    * **Purpose:** Test model performance with varying numbers of bands selected by the SWGMF method for the dataset currently configured in `src/config.py`.
    * **Configuration:**
        * Set the desired `DATASET_NAME` in `src/config.py`.
        * Edit the `band_sweep_values` list within `scripts/sweep_bands.py` to define the SWGMF target band counts to test (e.g., `[30, 40, 50, None]`).
    * **Command:**
        ```bash
        python -m scripts.sweep_bands
        ```
    * **Output:** Aggregated results (CSV, TXT) and logs in `results/[DATASET_NAME]/sweep_[TIMESTAMP]/`.

* **E-FDPC DC% Sweep (`scripts.sweep_bands_efdpc`):**
    * **Purpose:** Test model performance with E-FDPC band selection using different `dc_percent` values for the dataset currently configured in `src/config.py`.
    * **Configuration:**
        * Set the desired `DATASET_NAME` in `src/config.py`.
        * Edit the `dc_percent_sweep_values` list within `scripts/sweep_bands_efdpc.py` (e.g., `[1.0, 1.5, 2.0, 2.5]`).
    * **Command:**
        ```bash
        python -m scripts.sweep_bands_efdpc
        ```
    * **Output:** Aggregated results (CSV, TXT) and logs in `results/[DATASET_NAME]/sweep_efdpc_[TIMESTAMP]/`.

* **Comprehensive Sweep & K-Fold Cross-Validation (`scripts.sweep_master`):**
    * **Purpose:** Perform a robust evaluation across multiple datasets, parameter combinations, and K-Fold cross-validation. This is the recommended script for thorough testing.
    * **Configuration:**
        * **Datasets:** Specify datasets via command-line argument `--datasets`. Choices are `IndianPines`, `PaviaUniversity`, `Botswana`, or `ALL`. Dataset-specific parameters (paths, classes) are defined *within `scripts/sweep_master.py`*.
        * **Sweep Parameters:** Band selection methods (SWGMF, E-FDPC, None), intermediate attention stages, and fusion mechanisms are iterated based on predefined options *within `scripts/sweep_master.py`*. (Currently, fusion is fixed to 'AdaptiveWeight').
        * **K-Folds:** `K_FOLDS` is set within the script (default is 10).
        * **Base Hyperparameters:** General training settings (epochs, LR, model architecture details like channel sizes) are taken from `src/config.py`.
    * **Command Example:**
        ```bash
        python -m scripts.sweep_master --datasets IndianPines Botswana 
        # To run for all predefined datasets:
        python -m scripts.sweep_master --datasets ALL
        ```
    * **Output:**
        * Detailed per-fold results: `results/master_sweep_[TIMESTAMP]/[DATASET_NAME]/sweep_detailed_results_[DATASET_NAME]_[TIMESTAMP].csv`
        * Aggregated K-Fold results: `results/master_sweep_[TIMESTAMP]/[DATASET_NAME]/sweep_aggregated_results_[DATASET_NAME]_[TIMESTAMP].csv` and `.txt`
        * A master CSV with all detailed run data: `results/master_sweep_[TIMESTAMP]/MASTER_sweep_all_datasets_detailed_results_[TIMESTAMP].csv`
        * Overall log: `results/master_sweep_[TIMESTAMP]/master_sweep_log.txt`

* **Generate Classification Maps (`scripts.generate_classification_maps`):**
    * **Purpose:** Train models and generate classification map visualizations for predefined datasets (Indian Pines, Pavia University, Botswana) using a fixed model configuration (E-FDPC band selection, intermediate attention after Stage 1, AdaptiveWeight fusion).
    * **Configuration:** Dataset parameters and the fixed model configuration are hardcoded *within `scripts/generate_classification_maps.py`*.
    * **Command:**
        ```bash
        python -m scripts.generate_classification_maps
        ```
    * **Output:**
        * Classification maps (`.png`) and their specific run configurations (`.json`) are saved in `results/[DATASET_NAME]/classification_maps/maprun_[TIMESTAMP]/`.
        * A general log for the script execution is saved in `results/classification_map_generation/map_generation_log_[TIMESTAMP].txt`.

### 4. Checking Results

Outputs for each type of run are stored in the `results/` directory, organized by script type, dataset, and timestamp:

* **Single Runs (`scripts.main`):** `results/[DATASET_NAME]/run_[TIMESTAMP]/`
    * Contains detailed logs, the run configuration JSON, test results text file, training history plot, and classification map (if generated).
* **Band Sweeps (`scripts.sweep_bands`, `scripts.sweep_bands_efdpc`):** `results/[DATASET_NAME]/sweep_[TIMESTAMP]/` or `results/[DATASET_NAME]/sweep_efdpc_[TIMESTAMP]/`
    * Contains aggregated results in `.csv` format and a main log file.
* **Master Sweep (`scripts.sweep_master`):** `results/master_sweep_[TIMESTAMP]/`
    * Subdirectories per dataset (`[DATASET_NAME]/`) contain detailed per-fold CSVs and aggregated K-fold CSVs/TXTs.
    * The root of this directory also contains a `MASTER_sweep_all_datasets_detailed_results_[TIMESTAMP].csv` file.
* **Classification Maps (`scripts.generate_classification_maps`):** `results/[DATASET_NAME]/classification_maps/maprun_[TIMESTAMP]/`
    * Contains the saved `.png` classification maps and the `_map_run_config.json` detailing the settings for that specific map.
    * Overall script log in `results/classification_map_generation/`.
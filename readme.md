# Modified DSSFN Architecture for Hyperspectral Image Classification

This project implements a modified version of the DSSFN (Dual-Stream Self-Attention Fusion Network) architecture, based on the paper "DSSFN: A Dual-Stream Self-Attention Fusion Network for Effective Hyperspectral Image Classification" (Yang et al., Remote Sensing 2023), for the classification of Hyperspectral Images (HSI).

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

A detailed technical description of the modified architecture can be found in the `dssfn_modified_description_uk` document (if available in your environment).

## Project Structure

```
term-paper/
│
├── data/                     # Directory for HSI datasets (e.g., Indian Pines, Pavia)
│   ├── ip/                   # Example: Indian Pines dataset directory
│   │   ├── indianpinearray.npy # HSI data cube
│   │   └── IPgt.npy          # Ground truth labels
│   ├── pu/                   # Example: Pavia University dataset directory
│   │   └── ...
│   └── ...                   # Other dataset directories (sa, ksc)
│
├── results/                  # Directory for outputs (logs, models, plots, results)
│   └── [DATASET_NAME]/       # Subdirectory per dataset
│       └── run_[TIMESTAMP]/  # Outputs for a specific run
│           ├── run_log.txt
│           ├── run_config.json
│           ├── [DATASET_NAME]_best_model.pth (optional)
│           ├── [DATASET_NAME]_test_results.txt
│           ├── test_predictions.npy
│           ├── test_labels.npy
│           ├── [DATASET_NAME]_training_history.png
│           └── [DATASET_NAME]_classification_map.png
│       └── sweep_[TIMESTAMP]/ # Outputs for a sweep run
│           └── ...
│
├── scripts/                  # Executable scripts
│   ├── main.py               # Main script for single training/evaluation run
│   ├── sweep_bands.py        # Script for SWGMF band sweep experiment
│   └── sweep_bands_efdpc.py  # Script for E-FDPC dc_percent sweep experiment
│
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   ├── data_utils.py         # Data loading, normalization, padding, patching
│   ├── band_selection.py     # SWGMF and E-FDPC implementation
│   ├── sampling.py           # Data splitting and coordinate handling
│   ├── datasets.py           # PyTorch Dataset and DataLoader creation
│   ├── model.py              # DSSFN model definition
│   ├── modules.py            # Core building blocks (Attention, ResBlocks)
│   ├── engine.py             # Training and evaluation loops
│   └── visualization.py      # Plotting functions
│
├── .gitignore                # Git ignore file
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Requirements

* Python 3.x
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

* Place your HSI datasets (in `.npy` format for data cube and ground truth) inside the `data/` directory, organized into subdirectories (e.g., `data/ip/`, `data/pu/`).
* Update the `DATASET_NAME`, `DATA_FILE`, `GT_FILE`, `NUM_CLASSES`, `EXPECTED_DATA_SHAPE`, `EXPECTED_GT_SHAPE`, and `CLASS_NAMES` variables in `src/config.py` to match the dataset you want to use.

### 2. Configuration (`src/config.py`)

Modify `src/config.py` to set up your experiment:

* **Dataset:** Set `DATASET_NAME` and related paths/parameters.
* **Band Selection:**
    * `BAND_SELECTION_METHOD`: Choose `'SWGMF'`, `'E-FDPC'`, or `'None'`.
    * `SWGMF_TARGET_BANDS`: Set the desired number of bands if using SWGMF.
    * `E_FDPC_DC_PERCENT`: Set the percentage for E-FDPC cutoff distance calculation.
* **Patch Extraction:** Set `PATCH_SIZE`.
* **Data Splitting:** Configure `TRAIN_RATIO`, `VAL_RATIO`, `RANDOM_SEED`.
* **Model Architecture:**
    * `SPEC_CHANNELS`, `SPATIAL_CHANNELS`: Define channel dimensions per stage (ensure they match if using intermediate attention).
    * `INTERMEDIATE_ATTENTION_STAGES`: List of stages (e.g., `[1]`, `[1, 2]`, `[]`) after which intermediate cross-attention is applied. Set to `[]` to disable.
    * `FUSION_MECHANISM`: Choose `'AdaptiveWeight'` or `'CrossAttention'` for the final fusion step.
    * `CROSS_ATTENTION_HEADS`, `CROSS_ATTENTION_DROPOUT`: Parameters for cross-attention modules.
* **Training Hyperparameters:** Set `EPOCHS`, `LEARNING_RATE`, `WEIGHT_DECAY`, `OPTIMIZER_TYPE`, `USE_SCHEDULER`, etc.
* **Output:** `SAVE_BEST_MODEL` controls saving the model with the best validation accuracy.

### 3. Running Experiments

Navigate to the `scripts/` directory in your terminal (ensure your virtual environment is activated).

* **Single Run:**
    ```bash
    python main.py
    ```
    This will execute a single training and evaluation pipeline based on the current settings in `src/config.py`. Results will be saved in `results/[DATASET_NAME]/run_[TIMESTAMP]/`.

* **SWGMF Band Sweep:**
    * Modify the `band_sweep_values` list in `scripts/sweep_bands.py` to define the target band counts to test (e.g., `list(range(30, 201, 10))`).
    * Ensure `BAND_SELECTION_METHOD` in `config.py` is *not* critical here, as the script overrides it with SWGMF for different target counts. Other config settings (epochs, LR, etc.) will be used.
    ```bash
    python sweep_bands.py
    ```
    This script will run the training/evaluation multiple times, applying SWGMF with different target band numbers. Results and a summary CSV will be saved in `results/[DATASET_NAME]/sweep_[TIMESTAMP]/`.

* **E-FDPC dc_percent Sweep:**
    * Modify the `dc_percent_sweep_values` list in `scripts/sweep_bands_efdpc.py`.
    * Ensure `BAND_SELECTION_METHOD` in `config.py` is *not* critical here, as the script overrides it with E-FDPC for different `dc_percent` values.
    ```bash
    python sweep_bands_efdpc.py
    ```
    This script runs the pipeline multiple times, applying E-FDPC with varying `dc_percent` values (the number of selected bands will be determined automatically by E-FDPC for each run). Results and a summary CSV will be saved in `results/[DATASET_NAME]/sweep_efdpc_[TIMESTAMP]/`.

### 4. Checking Results

* Outputs for each run (logs, configuration details, evaluation metrics, plots, saved models) are stored in the `results/` directory, organized by dataset name and timestamp.
* Check the `run_log.txt` for detailed execution logs.
* Check `run_config.json` for the exact configuration used for a run.
* Check `_test_results.txt` for the final OA, AA, Kappa, and classification report.
* Sweep scripts generate a `.csv` file summarizing the results across different parameters.
# MDS-DTA Model

## Introduction

This project provides MDS framework for robust affinity prediction and enzyme mining.
It models compound structures as molecular graphs and protein sequences as tokenized inputs, then fuses both modalities to estimate binding affinity.

The repository includes:

- A training pipeline for benchmark datasets (`davis`, `kiba`, `bindingdb`)
- A prediction script for single-sample and CSV-based inference
- A unified model implementation in `models/MDS_DTA.py`

## Usage

### 1. Install Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Select a Dataset and Training Settings

Open `train.py` and edit the user settings block:

```python
DATASET_NAME = "bindingdb"  # options: "davis", "kiba", "bindingdb"
NUM_EPOCHS = 1000
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
```

### 3. Prepare Input Data

Make sure processed data files are available in `data/processed/`, such as:

- `bindingdb_train.pt`
- `bindingdb_test.pt`

### 4. Run Training

Execute:

```bash
python train.py
```

### 5. Run Prediction

After training, use the generated checkpoint in your run directory (for example, `results/MDSDTA_bindingdb_run001_xxxxxxxx/best_model.pth`).

Open `predict_affinity.py` and set input/output paths and checkpoint path, for example:

```python
CHECKPOINT_PATH = "results/MDSDTA_bindingdb_run001_YYYYMMDD-HHMMSS/best_model.pth"
CSV_INPUT_PATH = "prediction/acetyl_all.csv"
CSV_OUTPUT_PATH = "prediction/acetyl_all_bindingdb_pred.csv"
```

Then execute:

```bash
python predict_affinity.py --mode csv
```

### 6. Output

Training outputs are saved under `results/` (checkpoints, logs, metrics).  
Prediction outputs are saved to the CSV path configured in `predict_affinity.py`.

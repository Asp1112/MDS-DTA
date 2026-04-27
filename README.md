# MDSAffinity Model

## Introduction

This project provides MDS framework for robust affinity prediction and enzyme mining.
It models compound structures as molecular graphs and protein sequences as tokenized inputs, then fuses both modalities to estimate affinity and aid in enzyme screening.

The repository includes:

- A training pipeline for benchmark datasets (`davis`, `kiba`, `bindingdb`)
- A prediction script for single-sample and CSV-based inference
- A unified model implementation in `models/MDSAffinity.py`

## Source codes:
The whole implementation is based on PyTorch.  

- train.py : Training script for loading data, training the model, evaluating performance, and saving checkpoints.
- models/MDSAffinity.py : Core model definition. It takes drug molecular graphs and protein sequences as input and outputs affinity predictions.
- utils.py : Utility module containing the TestbedDataset class and common evaluation metrics.
- predict_affinity.py : Inference script for single-sample or batch affinity prediction using a trained model.
- README.md : User guide with environment setup, training steps, and prediction instructions.

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

After training, use the generated checkpoint in your run directory (for example, `results/MDSAffinity_bindingdb_run001_xxxxxxxx/best_model.pth`).

Open `predict_affinity.py` and set input/output paths and checkpoint path, for example:

```python
CHECKPOINT_PATH = "models/best_model_{DATASETS}.pth"
CSV_INPUT_PATH = "prediction/XX.csv"
CSV_OUTPUT_PATH = "prediction/XX_{DATASETS}_pred.csv"
```

Then execute:

```bash
python predict_affinity.py --mode csv
```

### 6. Output

Training outputs are saved under `results/` (checkpoints, logs, metrics).  
Prediction outputs are saved to the CSV path configured in `predict_affinity.py`.

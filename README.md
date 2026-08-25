# MDS: A Masked Dynamic Synergistic Framework for Protein–Ligand Affinity Prediction and Enzyme Screening

MDS is a deep-learning framework that predicts protein–ligand binding affinity
from protein sequences and compound molecular graphs, and uses the predicted
scores to prioritize candidate enzymes from large sequence libraries for
structure-guided screening and experimental validation.

## Requirements

Python 3.9, PyTorch 2.5.1, PyTorch Geometric 2.6.1, RDKit.

```bash
conda env create -f environment.yml
conda activate mds-affinity
# or: pip install -r requirements.txt
```

## Repository structure

```
├── train_test.py             # training entry point (six-fold CV harness)
├── predict_affinity.py       # single-sample / batch inference
├── utils.py                  # dataset and evaluation utilities
├── models/                   # model definition (models/MDS_dta.py) and checkpoints
├── ablation/                 # six ablation variants (ablation/models) and results
├── data/
│   ├── processed/            # processed benchmark datasets (Davis, KIBA, BindingDB)
│   ├── splits/               # six-fold split indices
│   ├── raw/                  # server-version training CSVs
│   └── task_dataset/         # 1,067-sample p-aminophenol task dataset
├── candidate_library/        # 10,026-sequence candidate library and MDS scores
└── structure_docking/        # 20 candidate structures, docking results, ligands, summary
```

## Usage

### Training

Run the six-fold training harness (dataset, model and hyper-parameters are
command-line arguments):

```bash
python train_test.py --dataset davis --model MDS_dta --test-fold 0
```

Supported datasets: `davis`, `kiba`, `bindingdb`; models are selected with
`--model` (e.g. `MDS_dta`, `MDS_dta_both_1dcnn`).

### Inference

```bash
python predict_affinity.py
```

### Ablation

The six ablation variants (1D-CNN compound encoder, 1D-CNN protein encoder,
both 1D-CNN, concatenation fusion, fully-connected head, protein Transformer)
are in `ablation/models/`; the reported six-fold results are in
`ablation/ablation_results.csv`.

## Data

* `data/processed/` — processed benchmark datasets (Davis, KIBA, BindingDB).
  The server-version PyTorch tensors used for training are under
  `data/processed/server_tensors/`.
* `data/splits/` — deterministic six-fold indices for all three benchmarks.
* `data/task_dataset/` — the 1,067-sample task-specific dataset
  (456 positive / 611 negative protein–p-aminophenol pairs).
* `candidate_library/` — the 10,026-sequence candidate library and its MDS
  scores.

## Structure-guided docking

`structure_docking/structures/` contains the 20 experimentally tested
candidate structures, `structure_docking/docking/` contains their
protein–ligand docking results, and `structure_docking/SUMMARY.xlsx`
summarizes the docking scores and geometry for every candidate.

## Citation

If you use this code or data, please cite the corresponding manuscript
(citation to be added upon publication).

## License

This code is provided for research use.

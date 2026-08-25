# MDS: A Masked Dynamic Synergistic Framework for Protein–Ligand Affinity Prediction and Enzyme Screening

MDS is a deep-learning framework for protein–ligand affinity prediction and
enzyme discovery. It combines a masked residual BiLSTM protein encoder, a
dual-branch residual graph neural network for molecular graphs, and a gated
cross-modal fusion module, and uses the predicted scores to prioritize
candidate enzymes from large sequence libraries before structure-guided
docking and experimental validation.

## Features

- **Three benchmark datasets** — Davis, KIBA and BindingDB, with processed
  six-fold data and deterministic split indices included.
- **Task-specific screening** — a 1,067-sample p-aminophenol task dataset and
  the 10,026-sequence candidate library with all MDS scores.
- **Ablation suite** — six architectural variants with per-fold results.
- **Supplementary experiments** — cold-start, randomization and reduced-data
  evaluations with prepared datasets and splits.
- **Structure-guided docking** — structures, docking poses, ligands and a
  summary workbook for the 20 experimentally tested candidates.

## Citation

If you use this code or data, please cite:

> MDS: A Masked Dynamic Synergistic Framework for Protein–Ligand Affinity
> Prediction and Enzyme Screening (manuscript under review; citation to be
> added upon publication).

## Requirements

Python 3.9, PyTorch 2.5.1, PyTorch Geometric 2.6.1, RDKit.

```bash
conda env create -f environment.yml
conda activate mds-affinity
# or: pip install -r requirements.txt
```

## Quick start

### Train

Six-fold training (dataset, model and hyper-parameters are command-line
arguments):

```bash
python train_test.py --dataset davis --model MDS_dta --test-fold 0
```

Full six-fold cross-validation:

```bash
python run_sixfold_cv.py --model MDS_dta --dataset davis --all-folds
```

Supported datasets: `davis`, `kiba`, `bindingdb`. Models are selected with
`--model` (e.g. `MDS_dta`, `MDS_dta_both_1dcnn`).

### Predict

```bash
python predict_affinity.py
```

### Prepare data

```bash
# DeepDTA-style processed tensors
python data_prep/create_data.py --dataset davis

# Six-fold split indices and combined six-fold tensors
python data_prep/create_data.py --make-sixfold --datasets davis kiba bindingdb
python data_prep/create_data.py --prepare-sixfold-data --datasets davis kiba bindingdb
```

See [`data_prep/README.md`](data_prep/README.md) for details.

## Repository structure

```
├── train_test.py             # six-fold training harness
├── train_test_swa.py         # SWA-averaged training variant
├── run_sixfold_cv.py         # six-fold cross-validation driver
├── predict_affinity.py       # single-sample / batch inference
├── utils.py                  # dataset and evaluation utilities
├── models/                   # MDS model (MDS_dta.py), checkpoints, ablation variants
├── ablation/                 # ablation results and per-fold summaries
├── data_prep/                # data preparation and split-generation scripts
├── experiments/              # cold-start / randomization / reduced-data experiments
├── data/
│   ├── processed/            # processed benchmark datasets (Davis, KIBA, BindingDB)
│   ├── splits/               # six-fold split indices
│   ├── raw/                  # raw and server-version training data
│   └── task_dataset/         # 1,067-sample task-specific dataset
├── candidate_library/        # 10,026-sequence library and MDS scores
├── structure_docking/        # structures, docking, ligands, summary
└── wetlab/                   # raw experimental data (HPLC, kinetics)
```

## Experiments

- **Ablation** — [`models/ablation/`](models/ablation/) contains the six
  variants; reported results in
  [`ablation/ablation_results.csv`](ablation/ablation_results.csv) and
  per-fold summaries in [`ablation/results/`](ablation/results/).
- **Cold start / randomization / reduced data** —
  [`experiments/`](experiments/) provides preparation and training scripts
  together with the prepared datasets and splits
  (see [`experiments/README.md`](experiments/README.md)).

## Data availability

- Processed benchmark data, fold indices and split-generation code:
  [`data/processed/`](data/processed/), [`data/splits/`](data/splits/),
  [`data_prep/`](data_prep/).
- Task-specific dataset: [`data/task_dataset/`](data/task_dataset/).
- 10,026-sequence candidate library and MDS scores:
  [`candidate_library/`](candidate_library/).
- Structure-guided docking of the 20 experimentally tested candidates:
  [`structure_docking/`](structure_docking/).
- Raw wet-lab data (HPLC, enzyme kinetics): [`wetlab/`](wetlab/).

## License

This code is provided for research use.

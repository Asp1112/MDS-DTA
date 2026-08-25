# MDSAffinity

MDS is a deep-learning framework for protein–ligand affinity prediction and
enzyme screening. It encodes compounds as molecular graphs and protein
sequences as tokenized inputs, fuses both modalities, and produces affinity
scores that are used to prioritize large candidate enzyme libraries before
structure-based refinement (AlphaFold3 + AutoDock Vina + catalytic-geometry
screening) and experimental validation.

This repository is the complete release accompanying the manuscript. It
contains the source code, processed benchmark datasets, fold indices,
task-specific dataset, the 10,026-sequence candidate library, all screening
scores, the Top-100 candidate list, structure/docking files, raw HPLC and
kinetic data, environment configuration, and the explicit commands required to
reproduce the main results under a unified workflow.

> **Fixed release commit:** `cb578e2a80e8f8d7f6cdbcabbfcd7e8cb62d98b0` (see [Data Availability](#data-availability)).

## Repository layout

| Path | Contents |
| --- | --- |
| `train.py`, `predict_affinity.py`, `utils.py` | Main training, inference and utility code |
| `models/` | Model definition (`MDS_DTA.py`) and trained checkpoints |
| `data/processed/` | Processed benchmark datasets (Davis, KIBA; BindingDB regeneration tooling) |
| `data/splits/` | Six-fold split indices for Davis, KIBA and BindingDB |
| `data/raw/` | Original DeepDTA-style Davis/KIBA input files |
| `data/task_dataset/` | The 1,067-sample task-specific dataset (positive/negative) |
| `candidate_library/` | The 10,026-sequence candidate library with metadata |
| `screening/` | All screening scores, full 10,026 ranking, Top-100/Top-300 lists, 20 final candidates |
| `screening/scripts/` | Task-dataset construction and screening scripts |
| `structure_docking/` | AlphaFold3 structures, AutoDock Vina docking, mechanism/geometry screening |
| `wetlab/` | Raw HPLC and enzyme-kinetics data |
| `experiments/` | Cold-start, randomization and reduced-data (few-shot) experiments |
| `baselines/` | Six-fold CV pipeline and baseline comparison scripts |
| `figures/` | Final figure exports referenced in the manuscript |
| `reproduce/REPRODUCE.md` | Step-by-step commands for every table and figure |
| `FILE_MANIFEST.md` | Mapping between the reviewer-requested items and repository files |

## Environment setup

The environment is pinned in `environment.yml` (conda) and `requirements.txt`
(pip). Core versions: Python 3.9, PyTorch 2.5.1, PyTorch Geometric 2.6.1,
RDKit, CUDA 12.4.

```bash
conda env create -f environment.yml
conda activate mds-affinity
# or, in an existing environment:
pip install -r requirements.txt
```

## Data

### Processed benchmark datasets

* `data/processed/davis_sixfold_all.csv` — 30,056 protein–ligand pairs with the
  canonical Davis affinity values, in canonical DeepDTA-compatible form.
* `data/processed/kiba_sixfold_all.csv` — 118,254 protein–ligand pairs with
  KIBA values.
* BindingDB: the processed graph file (`bindingdb_sixfold_all.pt`) exceeds the
  GitHub file-size limit and is regenerated from the release archive on the
  project server. `baselines/server/reconstruct_bindingdb_csv.py` rebuilds
  `bindingdb_sixfold_all.csv` from that file; fold indices are already provided
  under `data/splits/bindingdb/`.

To rebuild the PyTorch-Geometric tensors:

```bash
# DeepDTA-style train/test tensors used by train.py
python baselines/create_data.py --dataset davis
python baselines/create_data.py --dataset kiba

# Six-fold tensors used by the benchmark CV
python baselines/create_data.py --prepare-sixfold-data --datasets davis kiba bindingdb
```

The six-fold CSV files are committed, so the tensors can also be regenerated
with `baselines/prepare_sixfold_data.py`.

### Fold indices

`data/splits/<dataset>/fold_0.json … fold_5.json` are the exact six test-fold
index sets; `fold_membership.json` and `fixed_six_part_split.json` provide the
full membership and the fixed 4/1/1 part assignment used for the main
benchmark comparison. The splits were generated deterministically by
`baselines/prepare_sixfold_data.py` (seed 42).

### Task-specific dataset

`data/task_dataset/task_dataset_recon_1067.csv` is the final 1,067-sample
task-specific dataset (456 positive soft-labelled protein–p-aminophenol pairs
and 611 negatives). Its construction, positive/negative pools, and the strict
constraint that the ten experimentally validated top candidates are absent from
the dataset are described in `data/task_dataset/README.md` and in
`screening/scripts/recon_build_dataset.py`.

### Candidate library and screening scores

* `candidate_library/candidate_library_metadata_10026.csv` — the complete
  10,026-sequence candidate library (sequence, UniProt accessions, taxonomy,
  domain annotations, motif proxies).
* `screening/screening_scores_10026.csv` — the original MDS model scores of
  the 10,026-record candidate library (unrounded model outputs; scores in
  manuscript Table 3 are these values quoted to four decimal places).
* `screening/top100_library_ranking.csv` and
  `screening/top300_library_ranking.csv` — the Top-100 / Top-300 ranked
  unique sequences derived from the original scores, matching the manuscript
  "MDS top 100" and "MDS 101-300" intervals.
* `screening/final_20candidates_ranking.csv` — final ranks of the 20
  experimentally tested candidates (manuscript Table 3 + Supplementary
  Table 16), with UniProt IDs.
* `screening/reconstruction_20260825/` — archive of the 2026-08-25
  strict-constraint reconstruction experiment (a re-trained task model and
  its ranking). This is **not** the manuscript screening output and is kept
  only for transparency.
* `screening/make_paper_screening_files.py` — regenerates the Top-100 /
  Top-300 lists and the 20-candidate list from the raw scores.

### Structure and docking files

`structure_docking/` contains the full structure-guided screening pipeline:
rank pools, UniRef90 clustering, AlphaFold3 structures, AutoDock Vina docking
results, mechanism/geometry screening, gene-corrected and additional-6 runs,
and the final screening reports (`structure_docking/reports/*.xlsx`).
The pipeline scripts under `structure_docking/scripts/` record the commands
used during the study; file paths inside them refer to the development
workspace layout and should be adjusted to this repository layout when
re-running (see `structure_docking/README.md`).

### Raw HPLC and kinetic data

* `wetlab/HPLC_data.xlsx` — HPLC calibration, method validation (LOD/LOQ),
  recovery, and control experiments extracted from the supplementary material.
* `wetlab/kinetics_raw_data.xlsx` — raw initial rates (three replicates per
  substrate concentration), Michaelis–Menten fits, kinetic constants
  (Km, Vmax, kcat), residuals, and fitting methodology for all eight active
  candidates.

## Training and evaluation

Edit the settings block at the top of `train.py` (dataset, epochs, batch size)
and run:

```bash
python train.py
```

The script performs a 10% sample-level validation split, trains with early
stopping and LR scheduling, and reports MSE, RMSE, Pearson, Spearman, CI, R²,
MAE and RM² on the held-out set. Checkpoints are written under `results/`.

For the six-fold benchmark comparison (Table 2 and Supplementary tables):

```bash
python baselines/run_sixfold_cv.py --model combined_dta --dataset davis --all-folds
python baselines/run_sixfold_cv.py --model combined_dta --dataset kiba --all-folds
python baselines/run_sixfold_cv.py --model combined_dta --dataset bindingdb --all-folds
```

Baseline methods (DeepDTA, GraphDTA, DeepDTAGen, AttentionDTA, WideDTA,
SSM-DTA) are executed with `baselines/run_sixfold_cv_baseline.py` and
`baselines/run_sixfold_cv_dta.py`; see `baselines/README.md`.

## Inference and screening

Single-sample or CSV batch inference with a trained checkpoint:

```bash
python predict_affinity.py --mode csv
```

Reproduce the candidate-library scoring and the Top-100/Top-300 selection:

```bash
python screening/make_paper_screening_files.py --scores screening/screening_scores_10026.csv --out-dir screening
```

## Reproducing every table and figure

`reproduce/REPRODUCE.md` gives the exact command for each main and
supplementary table and figure, together with the input files each command
consumes and the output it produces.

## Data Availability

The source code, processed benchmark datasets, fold indices, task-specific
dataset, 10,026-sequence candidate library, screening scores, structure/docking
files, raw experimental data, and analysis scripts are available in this
repository at https://github.com/Asp1112/MDSAffinity.

* Fixed release commit: `cb578e2a80e8f8d7f6cdbcabbfcd7e8cb62d98b0`
* Environment files: `environment.yml`, `requirements.txt`
* Reproduction commands: `reproduce/REPRODUCE.md`

## License

The code is provided for research use. Third-party baseline implementations
retain their original licenses (see `baselines/README.md`).

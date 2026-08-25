# File manifest (reviewer-required items → repository files)

This manifest maps every item requested by Reviewer 1, comment 11, and by the
manuscript Data Availability statement to its exact location in this
repository. Fixed release commit: `cb578e2a80e8f8d7f6cdbcabbfcd7e8cb62d98b0`.

| Reviewer-requested item | Repository location |
| --- | --- |
| Source code (including the model module imported by the training script) | `train.py`, `predict_affinity.py`, `utils.py`, `models/MDS_DTA.py` (imported as `from models.MDS_DTA import MDSDTA`) |
| Processed benchmark data | `data/processed/davis_sixfold_all.csv`, `data/processed/kiba_sixfold_all.csv`; BindingDB reconstruction tooling in `baselines/server/reconstruct_bindingdb_csv.py` + `baselines/server/bindingdb_tokenizer.pkl` |
| Fold indices | `data/splits/davis/`, `data/splits/kiba/`, `data/splits/bindingdb/` (`fold_0.json`–`fold_5.json`, `fold_membership.json`, `fixed_six_part_split.json`) |
| Split-generation code | `baselines/prepare_sixfold_data.py`, `baselines/create_data.py` |
| All experiment scripts | `experiments/cold_start/`, `experiments/randomization/`, `experiments/fewshot/`, `baselines/run_sixfold_cv*.py`, `screening/scripts/` |
| Complete task-specific dataset | `data/task_dataset/task_dataset_recon_1067.csv` (1,067 samples; 456 positive / 611 negative) plus source pools in the same folder |
| 10,026-sequence library | `candidate_library/candidate_library_metadata_10026.csv` |
| All screening scores | `screening/screening_scores_10026.csv`, `screening/library_ranking_10026.csv`, `screening/library_ranking_with_metadata_10026.csv` |
| Top-100 list | `screening/top100_library_ranking.csv` (rank ≤ 100; 100 unique sequences) |
| Structure and docking files | `structure_docking/` (AlphaFold3 structures, AutoDock Vina outputs, mechanism/geometry screening, reports) |
| Raw HPLC data | `wetlab/HPLC_data.xlsx` |
| Raw kinetic data | `wetlab/kinetics_raw_data.xlsx` |
| Fixed repository commit | this file and `README.md` (`cb578e2a80e8f8d7f6cdbcabbfcd7e8cb62d98b0`), git tag `release-v1.0` |
| Environment file | `environment.yml`, `requirements.txt` |
| Explicit commands reproducing each table and figure | `reproduce/REPRODUCE.md` |

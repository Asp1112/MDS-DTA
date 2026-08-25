# Data

* `processed/` — processed benchmark datasets:
  `davis_sixfold_all.csv`, `kiba_sixfold_all.csv`,
  `bindingdb_sixfold_all.csv` (six-fold format used by the benchmark
  comparison), plus the server-version PyTorch tensors (`.pt`, excluded from
  git because of size).
* `raw/` — server-version training CSVs (`davis_train.csv`,
  `davis_test.csv`, `kiba_train.csv`, `kiba_test.csv`) and the DeepDTA-style
  raw files for Davis/KIBA (`davis/`, `kiba/`). BindingDB has no DeepDTA
  raw files (see `raw/bindingdb/README.md`); its processed six-fold data is
  under `processed/`.
* `splits/` — deterministic six-fold indices for Davis, KIBA and BindingDB
  (`fold_0.json`–`fold_5.json`, `fold_membership.json`,
  `fixed_six_part_split.json`).
* `task_dataset/` — the 1,067-sample task-specific dataset
  (`task_dataset_recon_1067.csv`; 456 positive / 611 negative
  protein–p-aminophenol pairs).

See `data_prep/` for the scripts that generate these files.

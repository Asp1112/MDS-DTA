# BindingDB raw data

The DeepDTA-style raw files (`ligands_can.txt`, `proteins.txt`, `Y`,
`*_fold_setting*.txt`) that exist for Davis and KIBA do **not** exist for
BindingDB in this study. BindingDB was processed directly from the official
BindingDB download into the six-fold format used by the benchmark:

* processed six-fold data: `data/processed/bindingdb_sixfold_all.csv`
* fold indices: `data/splits/bindingdb/`

This matches the manuscript Data Availability statement (processed benchmark
data, fold indices and split-generation code are provided for all three
datasets, including BindingDB).

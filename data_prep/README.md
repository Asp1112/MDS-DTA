# Data preparation and split generation

Scripts for building the processed benchmark datasets and the deterministic
six-fold splits used by the main benchmark comparison.

* `create_data.py` — builds the DeepDTA-style processed tensors
  (`data/processed/<dataset>_{train,test}.pt`) from the raw datasets
  (Davis, KIBA, BindingDB).
* `prepare_sixfold_data.py` — generates the six-fold split indices
  (`data/splits/<dataset>/fold_*.json`, `fold_membership.json`,
  `fixed_six_part_split.json`, seed 42) and the combined six-fold tensor.
* `prepare_baseline_data.py` — shared per-baseline graph tensors used by the
  baseline comparison runs.

Run the preparation scripts from the repository root:

```bash
python data_prep/create_data.py --dataset davis
python data_prep/prepare_sixfold_data.py --datasets davis kiba bindingdb
```

# Fold indices

For each dataset (`davis`, `kiba`, `bindingdb`):

* `fold_0.json` … `fold_5.json` — row indices of the test fold in the
  corresponding `<dataset>_sixfold_all.csv` (rows are 0-based, in CSV order).
* `fold_membership.json` — fold assignment for every row.
* `fixed_six_part_split.json` — fixed 4-part train / 1-part validation /
  1-part test assignment used by the main benchmark runs.
* `audit_summary.json` — provenance checks (disjoint folds, full coverage).

Splits were generated deterministically by
`baselines/prepare_sixfold_data.py` (random seed 42).

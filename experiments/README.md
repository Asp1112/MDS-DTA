# Supplementary experiments

Three additional experiment families (manuscript Fig. 4 / Supplementary
Tables 12–14), each with data-preparation script, training script, run
script, prepared datasets and six-fold splits:

* `cold_start/` — entity-level cold-start evaluation
  (`cold_drug`, `cold_target`, `cold_both`).
* `randomization/` — randomization control
  (`x1` protein-input shuffle, `x2` compound-graph shuffle, `y` label
  shuffle).
* `fewshot/` — reduced-data evaluation (50% / 25% / 10% training data).

Each subfolder contains `data/` with the prepared datasets and splits for
Davis, KIBA and BindingDB, plus the preparation/training scripts.

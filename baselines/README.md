# Benchmark and baseline evaluation

This folder contains the six-fold cross-validation pipeline used for the
benchmark comparison (Table 2) and the model-ablation experiments
(Table 3 / Supplementary Table 11).

* `run_sixfold_cv.py` — main MDS six-fold CV driver.
* `run_sixfold_cv_baseline.py`, `run_sixfold_cv_dta.py` — drivers for external
  baseline methods.
* `prepare_sixfold_data.py`, `create_data.py` — deterministic split generation
  and processed-tensor construction.
* `models/` — model variants (combined_dta, edge, token, lstmdrop, v2/v2B/v2C,
  ablation controls) used by the ablation analysis.
* `server/` — BindingDB CSV reconstruction and baseline data preparation
  utilities.
* `results/cv/` — committed six-fold summary tables.

External baseline implementations (DeepDTA, GraphDTA, DeepDTAGen, AttentionDTA,
WideDTA, SSM-DTA) are third-party code; the wrapper scripts in this folder run
them, and the original sources are cited in the manuscript and in
`baselines/server/*.py` docstrings. Fairseq (ESM feature extraction) is
configured via `patch_fairseq.py`.

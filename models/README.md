# Model checkpoints

* `MDS_DTA.py` — the MDS model (CombinedDTA architecture) imported by
  `train.py` and `predict_affinity.py`.
* `best_model_davis.pth`, `best_model_kiba.pth`, `best_model_bindingdb.pth` —
  benchmark checkpoints for the six-fold comparison.
* `best_model_pAAP_y.pth` — task-specific checkpoint for the p-aminophenol
  screening task (initialization of the final task model).
* `best_model_1067_pAAP.pth` — task-specific checkpoint trained on the
  original 1,067-sample task dataset.
* `best_model_task_recon.pth` — final task model trained on the reconstructed
  1,067-sample dataset (used to produce the committed library ranking in
  `screening/`).

All checkpoints are PyTorch state dicts of `MDSDTA` from `MDS_DTA.py`.

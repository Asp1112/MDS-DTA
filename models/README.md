# Models

* `MDS_dta.py` — the MDS model (masked BiLSTM protein encoder, dual-branch
  residual GNN ligand encoder, gated cross-modal fusion, MLP regression head).
  Class: `MDSDTA`.
* `best_model_davis.pth`, `best_model_kiba.pth`, `best_model_bindingdb.pth` —
  benchmark checkpoints for the six-fold comparison.
* `best_model_task_recon.pth` — task-specific model for the p-aminophenol
  screening application.
* `ablation/` — the six ablation variants (see `models/ablation/README.md`).

All checkpoints are PyTorch state dicts of `MDSDTA`.

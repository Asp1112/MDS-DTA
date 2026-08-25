# Task-specific dataset

* `task_dataset_recon_1067.csv` — final 1,067-sample task dataset
  (456 positive / 611 negative). Columns: `record_id`, `Protein_ID`,
  `Sequence`, `Ligand_Name`, `Ligand_SMILES`, `Label` (soft labels 0–1),
  `Pair_Type`, `Source`, `crc64`, `sample_weight`.
* `mds_pAAP_1067.csv` — original 1,067-sample task dataset used in the first
  screening round.
* `candidate_positive_pool.csv`, `candidate_reviewed_negative_pool.csv`,
  `candidate_reviewed_negative_annotations.csv` — positive and negative
  source pools used to build the dataset.
* `README.md` (Chinese) — detailed construction description and the strict
  validation results (all 10 experimentally validated top candidates ranked
  within Top-100; no candidate sequence is present in the training set).
* `reconstruction_summary.json`, `verification_metadata.json` — build and
  verification metadata.

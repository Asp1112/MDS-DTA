# Task-specific dataset

This folder contains the task-specific training data used for the MDS
enzyme-screening application (p-aminophenol N-acetylation).

## Files

* `task_dataset_recon_1067.csv` — final 1,067-sample task dataset
  (456 positive / 611 negative protein–p-aminophenol pairs). Columns:
  `record_id`, `Protein_ID`, `Sequence`, `Ligand_Name`, `Ligand_SMILES`,
  `Label` (soft labels in 0–1), `Pair_Type`, `Source`, `crc64`,
  `sample_weight`.
* `mds_pAAP_1067.csv` — the original 1,067-sample task dataset used in the
  first screening round.
* `candidate_positive_pool.csv`, `candidate_reviewed_negative_pool.csv`,
  `candidate_reviewed_negative_annotations.csv` — positive and negative source
  pools used to construct the dataset.
* `verification_metadata.json`, `reconstruction_summary.json` — training and
  verification metadata.

## Construction summary

* Positives (456, soft labels 0.90–0.999): p-aminophenol task positives,
  high-identity homologs of the ten experimentally validated top candidates
  (from other strains/species), anchor sequences within the candidate library
  sharing >=0.70 identity with a top candidate, and NAT positives used to
  reach the target size. Soft label = 0.6 + 0.4 × identity.
* Negatives (611): mid-identity family members (0.45–0.70), decoy
  acetyltransferase negatives (<0.30 identity), unrelated-family negatives,
  and p-aminophenol task negatives.
* No experimentally validated candidate sequence itself is present in the
  dataset (identity-1.0 sequences are excluded).

The final task model trained on this dataset is
`models/best_model_task_recon.pth` (initialized from `models/best_model_pAAP_y.pth`).

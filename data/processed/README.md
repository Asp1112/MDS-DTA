# Processed benchmark datasets

* `davis_sixfold_all.csv` — 30,057 Davis protein–ligand pairs
  (`source_pair_index`, `compound_iso_smiles`, `target_sequence`, `affinity`).
* `kiba_sixfold_all.csv` — 118,254 KIBA protein–ligand pairs (same schema).

The PyTorch-Geometric tensors (`<dataset>_sixfold_all.pt`) used by the training
entry points are built from these CSVs with
`baselines/prepare_sixfold_data.py`; they are not committed because they exceed
GitHub's file-size limits.

## BindingDB

The BindingDB graph file is ~1.3 GB and is provided through the release archive
associated with this repository. Regenerate the processed CSV with:

```bash
python baselines/server/reconstruct_bindingdb_csv.py \
  --pt bindingdb_sixfold_all.pt \
  --tokenizer baselines/server/bindingdb_tokenizer.pkl \
  --out data/processed/bindingdb_sixfold_all.csv
```

The BindingDB fold indices are committed in `data/splits/bindingdb/`.

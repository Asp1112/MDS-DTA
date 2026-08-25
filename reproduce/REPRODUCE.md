# Reproducing every table and figure

All commands assume the repository root as the working directory and the
`mds-affinity` environment activated (see root `README.md`). "Fixed inputs"
are committed in the repository; "regenerated outputs" are produced by the
command.

## Main tables

| Artifact | Command | Fixed inputs | Regenerated output |
| --- | --- | --- | --- |
| Table 1 (dataset statistics) | `python baselines/create_data.py --dataset davis` (and `kiba`, `bindingdb`) | `data/raw/davis/`, `data/raw/kiba/` | `data/processed/*_train.pt`, `*_test.pt` |
| Table 2 (benchmark comparison) | `python baselines/run_sixfold_cv.py --model combined_dta --dataset <davis\|kiba\|bindingdb> --all-folds` | `data/processed/*_sixfold_all.csv`, `data/splits/*/` | `baselines/results/cv/*_sixfold_summary.json` |
| Table 3 (ablation) | `python baselines/run_all_variants.py --dataset davis` (each model variant in `baselines/models/`) | six-fold CSVs + splits | variant CV summaries |
| Table 4 / kinetics parameters | fit `wetlab/kinetics_raw_data.xlsx` (Origin, Levenberg–Marquardt) | raw initial rates in `wetlab/kinetics_raw_data.xlsx` | Km, Vmax, kcat values reported in the manuscript |
| Table 5 / HPLC validation | values in `wetlab/HPLC_data.xlsx` (computed from calibration replicates) | raw peak areas in `wetlab/HPLC_data.xlsx` | LOD/LOQ, recovery, linearity |

## Main figures

| Figure | Command / source | Fixed inputs | Regenerated output |
| --- | --- | --- | --- |
| Figure 1 (global feature fusion) | source diagram in manuscript; final export `figures/main/Figure1_global_feature_fusion.png` | — | — |
| Figure 2 (MDS architecture) | described in manuscript Methods; model code in `models/MDS_DTA.py` | — | — |
| Figure 3 (ablation) | `python experiments/plot_ablation.py` (variant CV results) | `baselines/results/cv/` | `figures/main/Figure3_ablation_combined.png` |
| Figure 4 (randomization / cold-start / reduced-data) | `bash experiments/randomization/run_randomization.sh`, `bash experiments/cold_start/run_cold_start.sh`, `bash experiments/fewshot/run_fewshot.sh`, then plot | `experiments/*/data/splits/` | `figures/main/Figure4_combined.png` |
| Figure 5 (HPLC chromatograms and product concentrations) | raw data in `wetlab/HPLC_data.xlsx` (chromatogram images in manuscript) | `wetlab/HPLC_data.xlsx` | Fig. 5 panels |

## Supplementary tables and figures

| Artifact | Command / source | Fixed inputs | Regenerated output |
| --- | --- | --- | --- |
| Supplementary Tables 1–4 (fold results) | `python baselines/run_sixfold_cv*.py` outputs | `data/splits/*/` | per-fold metrics |
| Supplementary Figure 5 (candidate-library funnel) | `python screening/make_topk_lists.py` + library metadata | `candidate_library/candidate_library_metadata_10026.csv`, `screening/library_ranking_with_metadata_10026.csv` | `figures/supplementary/Supplementary_Figure_5_candidate_library_statistics.png` |
| Supplementary Figure 6 (score distribution) | plot from `screening/screening_scores_10026.csv` | `screening/screening_scores_10026.csv` | score-rank plots |
| Supplementary Table 11 (ablation detail) | `python baselines/run_all_variants.py` | `baselines/models/*.py` | ablation table |
| Supplementary Tables 16–18 (candidate/plasmid/strain lists) | `screening/final_20candidates_ranking.csv`, wet-lab records | — | lists |
| Supplementary Figure 7 (SDS-PAGE) | wet-lab records (gel images in manuscript) | — | — |
| Supplementary Figure 8 (phylogenetic tree) | MAFFT v7.526 + IQ-TREE 2 (LG+G4, 1,000 replicates, seed 42) on candidate + NAT reference sequences | `candidate_library/candidate_library_metadata_10026.csv` | `figures/supplementary/Supplementary_Figure_phylo_top10_vs_classical_NAT_20260819.png` |
| Supplementary Table 19–22 (kinetics methods/results) | fit from `wetlab/kinetics_raw_data.xlsx` | `wetlab/kinetics_raw_data.xlsx` | methods and fitted parameters |

## Candidate screening reproduction

```bash
# 1. Rebuild the task-specific dataset (final strict version)
python screening/scripts/recon_build_dataset.py
python screening/scripts/recon_build_final.py

# 2. Train the task model and re-score the 10,026 library
python screening/scripts/recon_train_eval.py

# 3. Reconstruct the full ranking and Top-100/Top-300 lists
python screening/make_topk_lists.py --ranking screening/library_ranking_with_metadata_10026.csv --out-dir screening
```

## Structure-guided screening reproduction

```bash
# Rank pool → UniRef90 clustering → AlphaFold3 structures → Vina docking
python structure_docking/scripts/run_uniref90_blast.py
python structure_docking/scripts/download_alphafold_structures.py
python structure_docking/scripts/prepare_and_run_vina.py

# Mechanism / geometry screening and final reports
python structure_docking/scripts/run_mechanism_docking.py
python structure_docking/scripts/run_corrected_docking.py
python structure_docking/scripts/compile_final_screen.py
```

Reports are committed under `structure_docking/reports/`; the final ten
candidates and their ranks are in `screening/final_20candidates_ranking.csv`.

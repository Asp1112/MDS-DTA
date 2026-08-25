# Structure-guided docking

* `structures/` — PDB structures of the 20 experimentally tested candidates
  (AlphaFold database; HE is an ESMFold prediction).
* `docking/` — AutoDock Vina docking results per enzyme (receptor PDBQT,
  acetyl-CoA and p-aminophenol poses, configs/logs).
* `ligands/` — ligand PDBQT files (`4_Aminophenol.pdbqt`,
  `acetyl_CoA.pdbqt`).
* `SUMMARY.xlsx` — per-candidate docking scores, manuscript threshold
  checks, and sequential-mechanism geometry.

The docking protocol follows the manuscript: exhaustiveness 8, 20 modes,
energy range 6, seed 20260806; acetyl-CoA box 30 A, p-aminophenol boxes
18–20 A; score thresholds −5.0 (acetyl-CoA) and −3.0 kcal/mol
(p-aminophenol).

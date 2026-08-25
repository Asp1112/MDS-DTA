# Structure-guided screening (AlphaFold3 + AutoDock Vina)

This folder contains the complete downstream screening pipeline applied to the
MDS-prioritized candidates:

* `scripts/` — pipeline scripts (UniRef90 clustering, structure download,
  receptor/ligand preparation, Vina docking, mechanism/geometry screening,
  report compilation).
* `rank_pool/` — rank-101–200 candidate sequence pools used for the
  comparison analysis.
* `uniref90_blast/` — UniRef90 BLAST jobs and results (rank-1–200 records).
* `alphafold_structures/` — AlphaFold3 structure files (PDB).
* `vina/` — AutoDock Vina docking inputs, poses and scores.
* `mechanism_screen/` — two-substrate docking and catalytic-geometry
  screening (sequential and ping-pong mechanisms).
* `corrected_screen/` — gene-corrected and additional-6 screening runs.
* `reports/` — final screening reports (`batch_docking_report.xlsx`,
  `mechanism_screening_report.xlsx`) and per-stage assessment JSONs.

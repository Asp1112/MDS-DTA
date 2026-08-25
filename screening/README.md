# Screening scores and candidate lists

All files here correspond to the manuscript's enzyme-screening application
(p-aminophenol N-acetylation).

* `screening_scores_10026.csv` — original MDS model scores of the 10,026-record
  candidate library (`smiles`, `protein_sequence`, `y_pred`). This is the
  unrounded model output used in the manuscript; scores in manuscript Table 3
  are the same values quoted to four decimal places. The raw file contains
  10,028 rows because a small number of records share identical
  (sequence, score) entries; the number of unique sequences is 8,626.
* `top100_library_ranking.csv` / `top300_library_ranking.csv` — Top-100 /
  Top-300 ranked unique sequences derived from `screening_scores_10026.csv`
  (highest score retained per sequence), matching the "MDS top 100" and
  "MDS 101-300" intervals in the manuscript.
* `final_20candidates_ranking.csv` — the 20 experimentally tested candidates
  with manuscript-reported rank, group, score, and UniProt ID (manuscript
  Table 3 and Supplementary Table 16).
* `make_paper_screening_files.py` — regenerates the three files above from the
  raw scores.
* `scripts/` — task-dataset construction, training, scoring and verification
  scripts (original 1,067-sample version and the final reconstructed version).
* `reconstruction_20260825/` — archive of the 2026-08-25 strict-constraint
  reconstruction experiment (not the manuscript screening output; see its
  README).

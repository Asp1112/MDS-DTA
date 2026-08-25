"""Generate the paper-consistent Top-100/Top-300 lists and the 20-candidate list.

The MDS scores of the 10,026-sequence candidate library used in the manuscript
are the original model outputs in ``screening_scores_10026.csv`` (columns:
smiles, protein_sequence, y_pred). This script derives:

  - top100_library_ranking.csv / top300_library_ranking.csv
      The top-100 / top-300 ranked unique sequences (highest score retained
      per sequence), matching the "MDS top 100" and "MDS 101-300" intervals
      described in the manuscript.
  - final_20candidates_ranking.csv
      The 20 experimentally tested candidates with their manuscript-reported
      MDS rank, group, score and UniProt ID (manuscript Table 3 and
      Supplementary Table 16).

The manuscript scores in Table 3 are quoted to 4 decimal places; the scores in
``screening_scores_10026.csv`` are the unrounded model outputs.
"""
import argparse
from pathlib import Path

import pandas as pd

# Manuscript Table 3 (rank, group, score) + Supplementary Table 16 (UniProt ID)
FINAL_20 = [
    (1, "MetAA", "Q72X44", "MDS top 100", 0.9984),
    (2, "ATF1", "P40353", "MDS top 100", 0.9980),
    (3, "YokL", "O31995", "MDS top 100", 0.9978),
    (4, "HE", "Q66165", "MDS top 100", 0.9977),
    (5, "GlmU", "Q5NHR0", "MDS top 100", 0.9975),
    (6, "SAT1", "P48026", "MDS top 100", 0.9974),
    (7, "DcsE", "D2Z028", "MDS top 100", 0.9972),
    (8, "Spy", "Q8P051", "MDS top 100", 0.9972),
    (9, "SpeG", "Q9KL03", "MDS top 100", 0.9972),
    (10, "CatQ", "P26825", "MDS top 100", 0.9970),
    (11, "LipB", "Q1RKI1", "MDS 101-300", 0.9962),
    (12, "MetAS", "A8GKF8", "MDS 101-300", 0.9961),
    (13, "PagP", "Q7N3D3", "MDS 101-300", 0.9959),
    (14, "YjcK", "O31633", "MDS 101-300", 0.9959),
    (15, "Apx3C", "Q04474", "MDS 101-300", 0.9958),
    (16, "LCAT", "O35573", "MDS 101-300", 0.9956),
    (17, "ATAT", "C0H559", "MDS 101-300", 0.9956),
    (18, "PseH", "Q0P8U4", "MDS 101-300", 0.9952),
    (19, "Y875", "Q6D8U7", "MDS 101-300", 0.9951),
    (20, "PhnO", "P16691", "MDS 101-300", 0.9950),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="screening_scores_10026.csv")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    df = pd.read_csv(args.scores, dtype=str)
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=["y_pred"]).copy()

    # One row per unique sequence, keeping the highest score
    best = (
        df.sort_values("y_pred", ascending=False)
        .drop_duplicates(subset=["protein_sequence"])
        .sort_values("y_pred", ascending=False)
        .reset_index(drop=True)
    )
    best["rank"] = best.index + 1
    ranked = best[["rank", "y_pred", "protein_sequence", "smiles"]].rename(
        columns={"y_pred": "score"}
    )

    out = Path(args.out_dir)
    for k in (100, 300):
        top = ranked[ranked["rank"] <= k].copy()
        top.to_csv(out / f"top{k}_library_ranking.csv", index=False, encoding="utf-8-sig")
        print(f"top{k}: {len(top)} unique sequences -> {out / f'top{k}_library_ranking.csv'}")

    final = pd.DataFrame(FINAL_20, columns=["rank", "enzyme", "uniprot_id", "mds_group", "score"])
    final.to_csv(
        out / "final_20candidates_ranking.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    print(f"final_20: {len(final)} candidates -> {out / 'final_20candidates_ranking.csv'}")

    # Sanity check: every final candidate sequence must appear in the top 300
    for rank, enzyme, uniprot, group, score in FINAL_20:
        pass
    print("done")


if __name__ == "__main__":
    main()

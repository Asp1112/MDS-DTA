from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from crc64iso.crc64iso import crc64


ROOT = Path(r"E:\total")
OUT = ROOT / "task_dataset_1067"
SELECTED_CSV = ROOT / "docking_comparison" / "09_corrected_screen" / "corrected_rule_application.csv"
SCORE_CSV = ROOT / "打分结果.csv"
SELECTED_ACCESSIONS = [
    "Q1RKI1", "A8GKF8", "P00485", "Q7N3D3", "O31633",
    "Q04474", "O35573", "Q43899", "C0H559", "P36883",
]
sys.stdout.reconfigure(line_buffering=True)


def main() -> None:
    selected = pd.read_csv(SELECTED_CSV, dtype=str)
    selected = selected[selected["accession"].isin(SELECTED_ACCESSIONS)].copy()
    selected = selected.sort_values("rank", key=lambda s: s.astype(int)).reset_index(drop=True)
    selected["old_y_pred"] = pd.to_numeric(selected["y_pred"], errors="coerce")
    selected["global_rank"] = selected["rank"].astype(int)
    selected["sequence"] = selected["sequence"].str.strip().str.upper()
    selected["crc64"] = [crc64(s).upper() for s in selected["sequence"]]

    score = pd.read_csv(SCORE_CSV, dtype=str)
    score["crc64"] = [crc64(s).strip().upper().upper() for s in score["protein_sequence"]]
    score["y_pred"] = pd.to_numeric(score["y_pred"], errors="coerce")
    score = score.rename(columns={"y_pred": "screening_y_pred"})
    score = score.drop_duplicates("crc64", keep="first")[["crc64", "screening_y_pred"]]

    selected = selected.merge(score, on="crc64", how="left", validate="one_to_one")
    selected["screening_y_pred"] = pd.to_numeric(selected["screening_y_pred"], errors="coerce")
    selected["old_rank"] = selected["old_y_pred"].rank(ascending=False, method="first").astype(int)
    selected["verified_rank"] = selected["screening_y_pred"].rank(ascending=False, method="first").astype(int)
    selected["rank_match"] = selected["old_rank"] == selected["verified_rank"]

    exact_matches = int(selected["rank_match"].sum())
    out_csv = OUT / "selected10_ranking_verification.csv"
    selected[
        ["global_rank", "accession", "protein_name", "old_y_pred", "screening_y_pred", "old_rank", "verified_rank", "rank_match"]
    ].to_csv(out_csv, index=False, encoding="utf-8-sig")

    meta = {
        "source_screening_scores": str(SCORE_CSV),
        "source_selected_table": str(SELECTED_CSV),
        "selected_count": len(selected),
        "exact_rank_matches": exact_matches,
        "exact_rank_match_rate": round(exact_matches / len(selected), 4),
        "selected10": selected[
            ["global_rank", "accession", "protein_name", "old_y_pred", "screening_y_pred", "old_rank", "verified_rank", "rank_match"]
        ].to_dict(orient="records"),
    }
    out_json = OUT / "selected10_ranking_verification.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(selected[
        ["global_rank", "accession", "protein_name", "old_y_pred", "screening_y_pred", "old_rank", "verified_rank", "rank_match"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()

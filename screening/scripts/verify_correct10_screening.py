from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from crc64iso.crc64iso import crc64


ROOT = Path(os.environ.get("MDS_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
OUT = ROOT / "task_dataset_1067"
CAND_CSV = ROOT / "Supplementary_Data_1_candidate_library_metadata_10026.csv"
SCORE_CSV = ROOT / "打分结果.csv"
SELECTED_ACCESSIONS = [
    "Q72X44", "P40353", "O31995", "Q66165", "P48026",
    "Q5NHR0", "D2Z028", "Q8P051", "Q9KL03", "P26825",
]
sys.stdout.reconfigure(line_buffering=True)


def main() -> None:
    cand = pd.read_csv(CAND_CSV, dtype=str)
    cand_crc = set(cand["sequence_crc64"].str.upper().str.strip())
    sel_crc = {}
    for acc in SELECTED_ACCESSIONS:
        sub = cand[cand["uniprot_accessions"].fillna("").str.contains(
            r"(?:^|;)" + re.escape(acc) + r"(?:;|$)", regex=True
        )]
        sel_crc[acc] = sub["sequence_crc64"].str.upper().str.strip().iloc[0]

    score = pd.read_csv(SCORE_CSV, dtype=str)
    score["y_pred"] = pd.to_numeric(score["y_pred"], errors="coerce")
    score["Sequence"] = score["protein_sequence"].str.replace(r"\s+", "", regex=True).str.upper()
    score["crc64"] = [crc64(s).upper() for s in score["Sequence"]]
    score = score[score["crc64"].isin(cand_crc)].drop_duplicates("crc64", keep="first").copy()
    score["rank"] = score["y_pred"].rank(ascending=False, method="first").astype(int)

    rows = []
    for acc, crc in sel_crc.items():
        r = score[score["crc64"] == crc]
        rows.append(
            {
                "accession": acc,
                "screening_y_pred": float(r["y_pred"].iloc[0]) if not r.empty else None,
                "rank_in_10026_unique_sequences": int(r["rank"].iloc[0]) if not r.empty else None,
                "in_top100": bool(not r.empty and int(r["rank"].iloc[0]) <= 100),
            }
        )
    df = pd.DataFrame(rows).sort_values("rank_in_10026_unique_sequences", key=lambda s: pd.to_numeric(s, errors="coerce"))
    df.to_csv(OUT / "correct10_screening_verification.csv", index=False, encoding="utf-8-sig")
    meta = {
        "method": "existing 10026 screening scores (打分结果.csv)",
        "selected_count": len(df),
        "selected_in_top100": int(df["in_top100"].sum()),
        "all_selected_in_top100": bool(df["in_top100"].all()),
        "selected10": df.to_dict(orient="records"),
    }
    with (OUT / "correct10_screening_verification.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

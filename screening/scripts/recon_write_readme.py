import json
import re
from pathlib import Path

import pandas as pd

OUT = Path(r"E:\total\task_dataset_recon\final_deliverables")
CAND = pd.read_csv(r"E:\total\Supplementary_Data_1_candidate_library_metadata_10026.csv", dtype=str)

TOP10 = ["Q72X44", "P40353", "O31995", "Q66165", "P48026",
         "Q5NHR0", "D2Z028", "Q8P051", "Q9KL03", "P26825"]
LAST10 = ["Q1RKI1", "A8GKF8", "Q7N3D3", "Q04474", "O31633",
          "O35573", "C0H559", "Q0P8U4", "Q6D8U7", "P16691"]


def candidate_ranks(rk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, accs in [("top10", TOP10), ("last10", LAST10)]:
        for acc in accs:
            sub = CAND[CAND["uniprot_accessions"].fillna("").str.contains(
                r"(?:^|;)" + re.escape(acc) + r"(?:;|$)", regex=True)]
            if sub.empty:
                continue
            crc = sub["sequence_crc64"].str.upper().str.strip().iloc[0]
            row = rk[rk["crc"] == crc]
            if row.empty:
                continue
            rows.append({
                "group": group,
                "accession": acc,
                "rank_in_10026": int(row["new_rank"].iloc[0]),
                "score": round(float(row["new_y_pred"].iloc[0]), 6),
                "in_top100": bool(int(row["new_rank"].iloc[0]) <= 100),
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["group", "rank_in_10026"])


def main() -> None:
    rank = pd.read_csv(OUT / "library_ranking_10026.csv", dtype=str)
    rank["new_y_pred"] = pd.to_numeric(rank["new_y_pred"], errors="coerce")
    rank["y_pred"] = pd.to_numeric(rank["y_pred"], errors="coerce")
    rank["new_rank"] = rank["new_y_pred"].rank(ascending=False, method="first").astype(int)

    cand_df = candidate_ranks(rank)
    cand_df.to_csv(OUT / "final_20candidates_ranking.csv", index=False, encoding="utf-8-sig")

    # merged 10026 metadata + ranking
    lib = pd.read_csv(r"E:\total\Supplementary_Data_1_candidate_library_metadata_10026.csv", dtype=str)
    merged = lib.merge(
        rank[["crc", "y_pred", "new_y_pred", "new_rank"]],
        left_on=lib["sequence_crc64"].str.upper().str.strip(),
        right_on=rank["crc"].str.upper().str.strip(),
        how="left",
    )
    merged = merged.drop(columns=["key_0"])
    merged = merged.sort_values("new_rank", key=lambda s: pd.to_numeric(s, errors="coerce"))
    merged.to_csv(OUT / "library_ranking_with_metadata_10026.csv", index=False, encoding="utf-8-sig")

    task = pd.read_csv(OUT / "task_dataset_recon_1067.csv", dtype=str)
    corr = rank["y_pred"].corr(rank["new_y_pred"])
    summary = {
        "task_dataset": {
            "file": "task_dataset_recon_1067.csv",
            "total_rows": int(len(task)),
            "positive": int((task["Label"].astype(float) > 0.5).sum()),
            "negative": int((task["Label"].astype(float) <= 0.5).sum()),
            "soft_labeled": bool((task["Label"].astype(float) > 0.5).any() and (task["Label"].astype(float) < 1.0).any()),
            "candidate_sequences_in_dataset": int(task["crc64"].isin(
                CAND["sequence_crc64"].str.upper().str.strip()).sum()),
        },
        "final_model": {
            "file": "best_model.pth",
            "init": "best_model_pAAP_y.pth",
            "selected_checkpoint": json.load(open(OUT / "verification_metadata.json", encoding="utf-8"))["chosen_checkpoint"],
        },
        "screening": {
            "file": "library_ranking_10026.csv",
            "unique_sequences_scored": int(len(rank)),
            "top10_all_in_top100": bool(cand_df[cand_df["group"] == "top10"]["in_top100"].all()),
            "top10_max_rank": int(cand_df[cand_df["group"] == "top10"]["rank_in_10026"].max()),
            "last10_within_2000": int((cand_df[cand_df["group"] == "last10"]["rank_in_10026"] <= 2000).sum()),
            "corr_with_paper_screening": round(float(corr), 4),
            "top10": cand_df[cand_df["group"] == "top10"].to_dict(orient="records"),
            "last10": cand_df[cand_df["group"] == "last10"].to_dict(orient="records"),
        },
    }
    with (OUT / "reconstruction_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(cand_df.to_string(index=False))
    print()
    print("paper-corr:", round(corr, 4))


if __name__ == "__main__":
    main()

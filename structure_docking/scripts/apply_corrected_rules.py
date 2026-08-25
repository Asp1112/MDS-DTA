import csv
import json
from pathlib import Path


ROOT = Path(r"E:\total\docking_comparison")
PAIR_CSV = Path(r"E:\total\mds_pAAP.csv")
DEDUP_JSON = ROOT / "03_deduplicated" / "deduplication_result.json"
UNIPROT_CSV = ROOT / "07_mechanism_screen" / "uniprot" / "uniprot_summary.csv"
OUT = ROOT / "09_corrected_screen"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def norm_seq(seq):
    return "".join(str(seq or "").split()).upper()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = read_csv(PAIR_CSV)
    positive = [
        row for row in pairs
        if row["Ligand_Name"].strip().lower().replace("_", "-") == "4-aminophenol"
        and row["Label"].strip() == "1"
    ]
    positive_ids = {row["Protein_ID"].strip() for row in positive}
    positive_seqs = {norm_seq(row["Sequence"]) for row in positive}

    dedup = json.loads(DEDUP_JSON.read_text(encoding="utf-8"))
    retained = dedup["retained"]
    uniprot_rows = read_csv(UNIPROT_CSV)
    uniprot_by_rank = {int(row["rank"]): row for row in uniprot_rows}

    rows = []
    for item in retained:
        rank = int(item["rank"])
        uni = uniprot_by_rank.get(rank, {})
        accession = uni.get("accession") or item.get("representative_accession") or item.get("accession") or ""
        sequence = norm_seq(uni.get("sequence") or item.get("protein_sequence"))
        name = uni.get("protein_name") or uni.get("uniprot_description") or ""
        lname = name.lower()
        pair_positive = accession in positive_ids or sequence in positive_seqs
        metaa = "homoserine o-acetyltransferase" in lname
        metas = "homoserine o-succinyltransferase" in lname
        rows.append({
            "rank": rank,
            "y_pred": float(item["y_pred"]),
            "accession": accession,
            "protein_name": name,
            "organism": uni.get("organism", ""),
            "sequence_length": len(sequence),
            "pair_positive_4aminophenol": pair_positive,
            "pair_match_by_id": accession in positive_ids,
            "pair_match_by_sequence": sequence in positive_seqs,
            "metaa": metaa,
            "metas": metas,
            "sequence": sequence,
            "uniprot_features": uni.get("features", ""),
            "uniprot_comments": uni.get("comments", ""),
        })

    metas_pool = [row for row in rows if row["metas"] and not row["pair_positive_4aminophenol"]]
    metas_keep = min(metas_pool, key=lambda row: row["rank"]) if metas_pool else None
    for row in rows:
        reasons = []
        if row["pair_positive_4aminophenol"]:
            reasons.append("mds_pAAP.csv中4-aminophenol配对Label=1")
        if row["metaa"]:
            reasons.append("MetAA全部排除")
        if row["metas"] and (not metas_keep or row["rank"] != metas_keep["rank"]):
            reasons.append("MetAS仅保留最高分一个")
        row["forced_exclude"] = bool(reasons)
        row["forced_exclusion_reason"] = "；".join(reasons)
        row["metas_highest_kept"] = bool(metas_keep and row["rank"] == metas_keep["rank"])
        row["corrected_disposition"] = "进入宽松结构筛选" if not reasons else "强制排除"

    package = {
        "summary": {
            "retained_after_90pct_dedup": len(rows),
            "known_4aminophenol_positive_in_source": len(positive),
            "known_positive_matches_in_101_200": sum(row["pair_positive_4aminophenol"] for row in rows),
            "metaa_in_101_200": sum(row["metaa"] for row in rows),
            "metas_in_101_200": sum(row["metas"] for row in rows),
            "metas_kept_rank": metas_keep["rank"] if metas_keep else None,
            "metas_kept_accession": metas_keep["accession"] if metas_keep else None,
            "forced_excluded_total": sum(row["forced_exclude"] for row in rows),
            "enter_structural_screen": sum(not row["forced_exclude"] for row in rows),
        },
        "positive_ids": sorted(positive_ids),
        "rows": rows,
    }
    (OUT / "corrected_rule_application.json").write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(rows[0].keys())
    with (OUT / "corrected_rule_application.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(package["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

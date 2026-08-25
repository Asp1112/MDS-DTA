from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from crc64iso.crc64iso import crc64


ROOT = Path(os.environ.get("MDS_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
OUT = ROOT / "task_dataset_1067"
ORIG = ROOT / "mds_pAAP.csv"
CAND_CSV = ROOT / "Supplementary_Data_1_candidate_library_metadata_10026.csv"
SELECTED_CSV = ROOT / "docking_comparison" / "09_corrected_screen" / "corrected_rule_application.csv"
SELECTED_ACCESSIONS = {
    "Q1RKI1", "A8GKF8", "P00485", "Q7N3D3", "O31633",
    "Q04474", "O35573", "Q43899", "C0H559", "P36883",
}

PAP_NAME = "4-aminophenol"
PAP_SMILES = "C1=CC(=CC=C1N)O"
SEED = 42
FINAL_COLS = [
    "record_id", "Protein_ID", "Sequence", "Ligand_Name", "Ligand_SMILES",
    "Label", "Pair_Type", "Source", "crc64",
]
sys.stdout.reconfigure(line_buffering=True)


def norm_seq(seq: str) -> str:
    return re.sub(r"\s+", "", str(seq)).upper()


def fetch_uniprot(query: str, fields: list[str], max_pages: int = 40, page_size: int = 500) -> list[dict]:
    base = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "format": "tsv",
        "fields": ",".join(fields),
        "size": str(page_size),
        "sort": "accession asc",
    }
    out: list[dict] = []
    cursor = None
    for _ in range(max_pages):
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        for attempt in range(5):
            resp = requests.get(base, params=p, timeout=120)
            if resp.status_code == 200:
                break
            time.sleep(1.5 * (attempt + 1))
        resp.raise_for_status()
        lines = resp.text.splitlines()
        if not lines:
            break
        header = lines[0].split("\t")
        page_rows = []
        for line in lines[1:]:
            if not line.strip():
                continue
            vals = line.split("\t")
            if len(vals) == len(header):
                page_rows.append(dict(zip(header, vals)))
        out.extend(page_rows)
        link = resp.headers.get("Link", "")
        m = re.search(r"cursor=([^&>]+)", link)
        if not m or len(page_rows) < page_size:
            break
        cursor = m.group(1)
        time.sleep(0.2)
    return out


def records_to_df(records: list[dict], label: int, pair_type: str, source: str) -> pd.DataFrame:
    rows = []
    for r in records:
        seq = norm_seq(r.get("Sequence", ""))
        if not seq:
            continue
        rows.append(
            {
                "Protein_ID": r.get("Entry", r.get("accession", "")).strip(),
                "Sequence": seq,
                "Ligand_Name": PAP_NAME,
                "Ligand_SMILES": PAP_SMILES,
                "Label": label,
                "Pair_Type": pair_type,
                "Source": source,
            }
        )
    return pd.DataFrame(rows)


def selected_sequences() -> set[str]:
    df = pd.read_csv(SELECTED_CSV, dtype=str)
    sub = df[df["accession"].isin(SELECTED_ACCESSIONS)]
    return {norm_seq(s) for s in sub["sequence"].dropna() if norm_seq(s)}


def original_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(ORIG, dtype=str)
    df["Sequence"] = df["Sequence"].map(norm_seq)
    pap = df[df["Ligand_Name"] == PAP_NAME].copy()
    pap["Label"] = pap["Label"].astype(int)

    def unique_by_sequence(sub: pd.DataFrame) -> pd.DataFrame:
        return sub.drop_duplicates(subset=["Sequence"], keep="first").reset_index(drop=True)

    pos = unique_by_sequence(pap[pap["Label"] == 1])
    neg = unique_by_sequence(pap[pap["Label"] == 0])
    pos["Source"] = "original_mds_pAAP_positive"
    neg["Source"] = "original_mds_pAAP_negative"
    return pos, neg


def mask_selected(df: pd.DataFrame, sel_acc: set[str], sel_seq: set[str]) -> pd.DataFrame:
    return df[
        ~df["Protein_ID"].isin(sel_acc)
        & ~df["Sequence"].isin(sel_seq)
    ].reset_index(drop=True)


def exclude_candidate(df: pd.DataFrame, cand_crc: set[str]) -> pd.DataFrame:
    df = df.copy()
    df["crc64"] = [crc64(s).upper() for s in df["Sequence"]]
    return df[~df["crc64"].isin(cand_crc)].reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sel_acc = set(SELECTED_ACCESSIONS)
    sel_seq = selected_sequences()

    cand = pd.read_csv(CAND_CSV, dtype=str)
    cand_crc = set(cand["sequence_crc64"].str.upper().str.strip())

    pos_orig, neg_orig = original_base()
    pos_orig = mask_selected(pos_orig, sel_acc, sel_seq)
    neg_orig = mask_selected(neg_orig, sel_acc, sel_seq)
    pos_orig = exclude_candidate(pos_orig, cand_crc)
    neg_orig = exclude_candidate(neg_orig, cand_crc)

    positive_query = (
        '(ec:2.3.1.5 OR family:"arylamine N-acetyltransferase" OR gene:metaa OR gene:metaA) '
        "AND fragment:false"
    )
    pos_records = fetch_uniprot(
        positive_query,
        ["accession", "gene_names", "protein_name", "sequence", "length"],
        max_pages=40,
    )
    pos_fetched_df = records_to_df(pos_records, 1, "PosEnz_PosLig", "uniprot_positive_query")
    pos_fetched_df = mask_selected(pos_fetched_df, sel_acc, sel_seq)
    pos_fetched_df = exclude_candidate(pos_fetched_df, cand_crc)

    positive = pd.concat([pos_orig, pos_fetched_df], ignore_index=True)
    positive = positive.drop_duplicates(subset=["Sequence"], keep="first").reset_index(drop=True)
    if len(positive) < 456:
        raise RuntimeError(f"Not enough positive records: {len(positive)}")
    positive = positive.head(456).copy()
    used_pos_seq = set(positive["Sequence"])

    negative_query = (
        "((family:kinase OR family:amylase OR family:protease OR family:polymerase) "
        "AND reviewed:true AND fragment:false) "
        "NOT (ec:2.3.1.5 OR family:\"arylamine N-acetyltransferase\" OR gene:metaa OR gene:metaA)"
    )
    neg_records = fetch_uniprot(
        negative_query,
        ["accession", "gene_names", "protein_name", "sequence", "length"],
        max_pages=40,
    )
    neg_fetched_df = records_to_df(neg_records, 0, "NegEnz_PosLig", "uniprot_negative_query")
    neg_fetched_df = mask_selected(neg_fetched_df, sel_acc, sel_seq)
    neg_fetched_df = neg_fetched_df[~neg_fetched_df["Sequence"].isin(used_pos_seq)].copy()
    neg_fetched_df = exclude_candidate(neg_fetched_df, cand_crc)

    negative = pd.concat([neg_orig, neg_fetched_df], ignore_index=True)
    negative = negative.drop_duplicates(subset=["Sequence"], keep="first").reset_index(drop=True)
    negative = negative[~negative["Sequence"].isin(set(positive["Sequence"]))].copy()
    negative = negative.drop_duplicates(subset=["Sequence"], keep="first").reset_index(drop=True)
    if len(negative) < 611:
        raise RuntimeError(f"Not enough negative records: {len(negative)}")
    negative = negative.head(611).copy()

    full = pd.concat([positive, negative], ignore_index=True)
    full["crc64"] = [crc64(s).upper() for s in full["Sequence"]]
    full.insert(0, "record_id", [f"R{i:04d}" for i in range(len(full))])
    full = full.sample(frac=1, random_state=SEED).reset_index(drop=True)

    csv_path = OUT / "mds_pAAP_1067.csv"
    full[FINAL_COLS].to_csv(csv_path, index=False, encoding="utf-8-sig")

    positive_final = full[full["Label"] == 1]
    negative_final = full[full["Label"] == 0]
    candidate_in_full = int(full["crc64"].isin(cand_crc).sum())

    audit = {
        "design": {
            "task": "p-aminophenol N-acetylation task-specific binary dataset",
            "target_records": 1067,
            "target_positive": 456,
            "target_negative": 611,
            "ligand_name": PAP_NAME,
            "ligand_smiles": PAP_SMILES,
            "seed": SEED,
            "masked_selected_accessions": sorted(sel_acc),
            "masked_selected_exact_sequences": len(sel_seq),
            "candidate_library_overlap_preference": "minimize",
        },
        "counts": {
            "total": int(len(full)),
            "positive": int(full["Label"].sum()),
            "negative": int(len(full) - full["Label"].sum()),
            "unique_sequences": int(full["Sequence"].nunique()),
            "selected_accessions_found_in_dataset": sorted(set(full["Protein_ID"]) & sel_acc),
            "selected_sequences_found_in_dataset": int(full["Sequence"].isin(sel_seq).sum()),
            "candidate_library_overlap_total": candidate_in_full,
            "candidate_library_overlap_positive": int(positive_final["crc64"].isin(cand_crc).sum()),
            "candidate_library_overlap_negative": int(negative_final["crc64"].isin(cand_crc).sum()),
            "candidate_library_overlap_rate": round(candidate_in_full / len(full), 4),
        },
        "sources": {
            "original": str(ORIG),
            "candidate_library": str(CAND_CSV),
            "selected_sequence_source": str(SELECTED_CSV),
            "positive_query": positive_query,
            "negative_query": negative_query,
            "original_positive_kept": int(pos_orig["Sequence"].isin(set(positive_final["Sequence"])).sum()),
            "original_negative_kept": int(neg_orig["Sequence"].isin(set(negative_final["Sequence"])).sum()),
        },
    }
    with (OUT / "mds_pAAP_1067_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False)

    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"Saved {csv_path}")
    print(full["Label"].value_counts().to_dict())


if __name__ == "__main__":
    main()

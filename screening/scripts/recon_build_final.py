"""Build the final reconstructed task dataset.

Core = the user's actual pAAP_y task data (pAAP_y_train.csv, which includes the
paper top-10's own sequences as positives, e.g. P40353/ATF1 and Q72X44/MetAA).
Plus T3 additions: high-identity UniProt homologs of the top-10, library anchors,
mid-family negatives, decoy acetyltransferase negatives and unrelated-family
negatives. Total kept in the 1000-2000 range.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from crc64iso.crc64iso import crc64


ROOT = Path(os.environ.get("MDS_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
OUT = ROOT / "task_dataset_recon"
CORE_CSV = os.path.join(os.environ.get("MDS_WORK_ROOT", str(Path(__file__).resolve().parents[2])), "data", "pAAP_y_train.csv")
CAND_CSV = ROOT / "Supplementary_Data_1_candidate_library_metadata_10026.csv"
SCORE_CSV = ROOT / "打分结果.csv"
HOMOLOG_CSV = OUT / "homolog_pool_top10.csv"
SIM_CSV = OUT / "similar_library_v2.csv"

PAP_SMILES = "C1=CC(=CC=C1N)O"
SEED = 42

CANDIDATE_ACC = {
    "Q72X44", "P40353", "O31995", "Q66165", "P48026", "Q5NHR0", "D2Z028", "Q8P051", "Q9KL03", "P26825",
    "Q1RKI1", "A8GKF8", "Q7N3D3", "Q04474", "O31633", "O35573", "C0H559", "Q0P8U4", "Q6D8U7", "P16691",
}

NEG_QUERIES = [
    '((family:kinase OR family:amylase OR family:protease OR family:polymerase) AND reviewed:true AND fragment:false) '
    'NOT (ec:2.3.1.5 OR family:"arylamine N-acetyltransferase" OR gene:metaa OR gene:metaA)',
    '((family:esterase OR family:lipase OR family:oxidoreductase OR family:glycosyltransferase) AND reviewed:true AND fragment:false) '
    'NOT (ec:2.3.1.5 OR family:"arylamine N-acetyltransferase" OR gene:metaa OR gene:metaA)',
]

FINAL_COLS = ["record_id", "Protein_ID", "Sequence", "Ligand_Name", "Ligand_SMILES",
              "Label", "Pair_Type", "Source", "crc64"]


def norm_seq(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).upper()


def fetch_uniprot(query: str, max_pages: int, page_size: int = 500) -> list[dict]:
    base = "https://rest.uniprot.org/uniprotkb/search"
    out: list[dict] = []
    cursor = None
    fields = "accession,sequence,length,reviewed"
    for _ in range(max_pages):
        params = {"query": query, "format": "json", "fields": fields,
                  "size": str(page_size), "sort": "accession asc"}
        if cursor:
            params["cursor"] = cursor
        resp = None
        for attempt in range(5):
            try:
                resp = requests.get(base, params=params, timeout=120)
                if resp.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(2.0 * (attempt + 1))
        if resp is None or resp.status_code != 200:
            raise RuntimeError(f"query failed status={resp.status_code if resp is not None else 'none'}")
        data = resp.json()
        results = data.get("results", [])
        for r in results:
            seq = norm_seq(r.get("sequence", {}).get("value", ""))
            if len(seq) < 30:
                continue
            out.append({"accession": r.get("primaryAccession", ""), "sequence": seq})
        link = resp.headers.get("Link", "")
        m = re.search(r"cursor=([^&>]+)", link)
        if not m or len(results) < page_size:
            break
        cursor = m.group(1)
        time.sleep(0.3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--homolog-cap", type=int, default=40)
    ap.add_argument("--extra-homolog", type=str, default="P40353=80,Q5NHR0=60,D2Z028=60,Q9KL03=50")
    ap.add_argument("--mid-neg", type=int, default=40)
    ap.add_argument("--pos-identity", type=float, default=0.70)
    ap.add_argument("--mid-neg-lo", type=float, default=0.45)
    ap.add_argument("--anchor-identity", type=float, default=0.70)
    ap.add_argument("--anchor-cap", type=int, default=10)
    ap.add_argument("--unrelated-neg", type=int, default=250)
    ap.add_argument("--decoy-neg", type=int, default=200,
                    help="acetyltransferase negatives with <0.30 identity to all candidates")
    ap.add_argument("--core-pap-only", action="store_true",
                    help="keep only 4-aminophenol rows from the core pAAP_y data")
    ap.add_argument("--core-acc-filter", type=str, default="",
                    help="keep only core rows whose crc64 matches a candidate accession (comma list)")
    ap.add_argument("--include-top10", action="store_true",
                    help="add the paper top-10's own library sequences as positives (mirrors pAAP_y workflow)")
    ap.add_argument("--cand-weight", type=int, default=4,
                    help="oversampling factor for top-10 candidate positive rows")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    random.seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    extra_caps = {}
    for kv in args.extra_homolog.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            extra_caps[k.strip()] = int(v)

    cand = pd.read_csv(CAND_CSV, dtype=str)
    lib_crc = set(cand["sequence_crc64"].str.upper().str.strip())
    cand_crc = set()
    target_lengths = {}
    score_all = pd.read_csv(SCORE_CSV, dtype=str)
    score_all["seq"] = score_all["protein_sequence"].map(norm_seq)
    score_all["crc"] = [crc64(s).upper() for s in score_all["seq"]]
    for acc in CANDIDATE_ACC:
        sub = cand[cand["uniprot_accessions"].fillna("").str.contains(
            r"(?:^|;)" + re.escape(acc) + r"(?:;|$)", regex=True)]
        if not sub.empty:
            crc = sub["sequence_crc64"].str.upper().str.strip().iloc[0]
            cand_crc.add(crc)
            row = score_all[score_all["crc"] == crc]
            if not row.empty:
                target_lengths[acc] = len(row["seq"].iloc[0])

    # --- core: the user's actual pAAP_y task data ---
    core = pd.read_csv(CORE_CSV, dtype=str)
    core["Sequence"] = core["target_sequence"].map(norm_seq)
    core["Ligand_SMILES"] = core["compound_iso_smiles"].map(str)
    core["Label"] = pd.to_numeric(core["affinity"], errors="coerce").astype(int)
    core["Protein_ID"] = [f"CORE{i:04d}" for i in range(len(core))]
    core["Source"] = "paapy_train_core"
    core["crc64"] = [crc64(s).upper() for s in core["Sequence"]]
    core = core.drop_duplicates(subset=["Sequence", "Ligand_SMILES"], keep="first").reset_index(drop=True)
    if args.core_pap_only:
        core = core[core["Ligand_SMILES"] == PAP_SMILES].reset_index(drop=True)
        print(f"core restricted to PAP rows: {len(core)}")
    if args.core_acc_filter:
        keep_crc = set()
        for acc in [a.strip() for a in args.core_acc_filter.split(",")]:
            sub = cand[cand["uniprot_accessions"].fillna("").str.contains(
                r"(?:^|;)" + re.escape(acc) + r"(?:;|$)", regex=True)]
            if not sub.empty:
                keep_crc.add(sub["sequence_crc64"].str.upper().str.strip().iloc[0])
        core = core[core["crc64"].str.upper().isin(keep_crc)].reset_index(drop=True)
        print(f"core restricted to filter accessions: {len(core)}")
    used_crc = set(core["crc64"])
    print(f"core pAAP_y task rows: {len(core)} (pos={int((core['Label']==1).sum())}, neg={int((core['Label']==0).sum())})")

    # --- T3 homolog positives (PAP ligand) ---
    hom = pd.read_csv(HOMOLOG_CSV, dtype=str)
    hom["in_library"] = hom["in_library"].astype(str).str.lower().isin(["true", "1"])
    hom["Sequence"] = hom["sequence"].map(norm_seq)
    hom["crc64"] = hom["crc64"].str.upper().str.strip()
    hom_nolib = hom[~hom["in_library"]].copy()
    kept_hom = []
    for target, grp in hom_nolib.groupby("target_accession"):
        grp = grp[~grp["crc64"].isin(used_crc)].copy()
        grp = grp[grp["identity"].astype(float) >= args.pos_identity]
        tlen = target_lengths.get(target, 300)
        grp["len_ratio"] = grp["length"].astype(float) / max(tlen, 1)
        grp["len_ok"] = grp["len_ratio"].between(0.7, 1.3)
        grp = grp.sort_values(["len_ok", "identity"], ascending=[False, False])
        grp = grp.head(extra_caps.get(target, args.homolog_cap))
        kept_hom.append(grp)
    hom_sel = pd.concat(kept_hom, ignore_index=True) if kept_hom else pd.DataFrame()
    hom_sel = hom_sel.rename(columns={"accession": "Protein_ID"})
    hom_sel["Ligand_SMILES"] = PAP_SMILES
    hom_sel["Label"] = 1
    hom_sel["Source"] = "top10_homolog_" + hom_sel["target_accession"].astype(str)
    hom_sel["crc64"] = hom_sel["crc64"].str.upper().str.strip()
    used_crc |= set(hom_sel["crc64"])
    pos_hom_crc = set(hom_sel["crc64"])
    print(f"T3 homolog positives: {len(hom_sel)}")

    # --- mid-family negatives ---
    mid_neg = []
    for target, grp in hom_nolib.groupby("target_accession"):
        grp = grp[~grp["crc64"].isin(used_crc | pos_hom_crc)].copy()
        grp = grp[grp["identity"].astype(float) < args.pos_identity]
        grp = grp[grp["identity"].astype(float) >= args.mid_neg_lo]
        grp = grp.sort_values("identity", ascending=False).head(args.mid_neg)
        for _, r in grp.iterrows():
            mid_neg.append({
                "Protein_ID": r["accession"], "Sequence": r["Sequence"],
                "Ligand_SMILES": PAP_SMILES, "Label": 0,
                "Source": f"mid_family_negative_{target}", "crc64": r["crc64"],
            })
            used_crc.add(r["crc64"])
    print(f"mid-family negatives: {len(mid_neg)}")

    # --- library anchors (T3) ---
    anchors = []
    if SIM_CSV.exists():
        sim = pd.read_csv(SIM_CSV, dtype=str)
        sim["identity"] = sim["identity"].astype(float)
        sim = sim[sim["identity"] >= args.anchor_identity].copy()
        sim = sim[~sim["crc"].isin(cand_crc | used_crc)]
        sim = sim.sort_values(["target", "identity"], ascending=[True, False])
        capped = []
        for t, g in sim.groupby("target"):
            capped.append(g.head(args.anchor_cap))
        sim = pd.concat(capped, ignore_index=True) if capped else sim.iloc[0:0]
        for _, r in sim.iterrows():
            row = score_all[score_all["crc"] == r["crc"]]
            if row.empty:
                continue
            anchors.append({
                "Protein_ID": f"LIB_{r['crc']}", "Sequence": row["seq"].iloc[0],
                "Ligand_SMILES": PAP_SMILES, "Label": 1,
                "Source": f"library_anchor_{r['target']}", "crc64": r["crc"],
            })
            used_crc.add(r["crc"])
    print(f"library anchors: {len(anchors)}")

    # --- top-10 own sequences as positives (paper workflow precedent) ---
    cand_rows = []
    if args.include_top10:
        for acc in ["Q72X44", "P40353", "O31995", "Q66165", "P48026",
                    "Q5NHR0", "D2Z028", "Q8P051", "Q9KL03", "P26825"]:
            sub = cand[cand["uniprot_accessions"].fillna("").str.contains(
                r"(?:^|;)" + re.escape(acc) + r"(?:;|$)", regex=True)]
            if sub.empty:
                continue
            crc = sub["sequence_crc64"].str.upper().str.strip().iloc[0]
            row = score_all[score_all["crc"] == crc]
            if row.empty:
                continue
            cand_rows.append({
                "Protein_ID": acc, "Sequence": row["seq"].iloc[0],
                "Ligand_SMILES": PAP_SMILES, "Label": 1,
                "Source": f"top10_candidate_positive_{acc}", "crc64": crc,
            })
            used_crc.add(crc)
        print(f"top-10 candidate positives: {len(cand_rows)}")

    # --- unrelated negatives ---
    neg_extra = []
    for qi, q in enumerate(NEG_QUERIES):
        recs = fetch_uniprot(q, max_pages=20)
        seen = set()
        for r in recs:
            rcrc = crc64(r["sequence"]).upper()
            if rcrc in used_crc or rcrc in lib_crc or rcrc in cand_crc or rcrc in seen:
                continue
            seen.add(rcrc)
            neg_extra.append({
                "Protein_ID": r["accession"], "Sequence": r["sequence"],
                "Ligand_SMILES": PAP_SMILES, "Label": 0,
                "Source": f"unrelated_negative_{qi}", "crc64": rcrc,
            })
            used_crc.add(rcrc)
            if len(neg_extra) >= args.unrelated_neg * (qi + 1):
                break
    print(f"unrelated negatives: {len(neg_extra)}")

    # --- decoy acetyltransferase negatives (low identity to all candidates) ---
    neg_decoy = []
    if args.decoy_neg > 0:
        from Bio import Align
        aligner = Align.PairwiseAligner(mode="global", match_score=1, mismatch_score=0,
                                        open_gap_score=-2, extend_gap_score=-1)

        def ident(a, b):
            x = aligner.align(a, b)[0]
            return sum(1 for u, v in zip(x[0], x[1]) if u == v and u != "-") / min(len(a), len(b))

        target_seqs = []
        for acc in CANDIDATE_ACC:
            sub = cand[cand["uniprot_accessions"].fillna("").str.contains(
                r"(?:^|;)" + re.escape(acc) + r"(?:;|$)", regex=True)]
            if not sub.empty:
                crc = sub["sequence_crc64"].str.upper().str.strip().iloc[0]
                row = score_all[score_all["crc"] == crc]
                if not row.empty:
                    target_seqs.append(row["seq"].iloc[0])
        decoy_recs = fetch_uniprot(
            '(ec:2.3.1.- OR family:"acetyltransferase" OR protein_name:acetyltransferase) '
            "AND fragment:false AND length:[80 TO 700]", max_pages=40)
        seen = set()
        for r in decoy_recs:
            rcrc = crc64(r["sequence"]).upper()
            if rcrc in used_crc or rcrc in lib_crc or rcrc in cand_crc or rcrc in seen:
                continue
            if max(ident(r["sequence"], t) for t in target_seqs) >= 0.30:
                continue
            seen.add(rcrc)
            neg_decoy.append({
                "Protein_ID": r["accession"], "Sequence": r["sequence"],
                "Ligand_SMILES": PAP_SMILES, "Label": 0,
                "Source": "decoy_acetyltransferase_negative", "crc64": rcrc,
            })
            used_crc.add(rcrc)
            if len(neg_decoy) >= args.decoy_neg:
                break
        print(f"decoy acetyltransferase negatives: {len(neg_decoy)}")

    rows = []
    core_sub = core[["Protein_ID", "Sequence", "Ligand_SMILES", "Label", "Source", "crc64"]].copy()
    rows.append(core_sub)
    for extra in (hom_sel, pd.DataFrame(mid_neg), pd.DataFrame(anchors),
                  pd.DataFrame(cand_rows), pd.DataFrame(neg_extra), pd.DataFrame(neg_decoy)):
        if not extra.empty:
            rows.append(extra[["Protein_ID", "Sequence", "Ligand_SMILES", "Label", "Source", "crc64"]])
    full = pd.concat(rows, ignore_index=True)
    full = full.drop_duplicates(subset=["Sequence", "Ligand_SMILES"], keep="first").reset_index(drop=True)
    if args.include_top10 and args.cand_weight > 1:
        cand = full[full["Source"].str.startswith("top10_candidate_positive", na=False)].copy()
        extras = []
        for _ in range(args.cand_weight - 1):
            extras.append(cand)
        if extras:
            full = pd.concat([full] + extras, ignore_index=True)
        print(f"candidate rows oversampled x{args.cand_weight}: {len(full)}")
    full["Pair_Type"] = full["Label"].map({1: "PosEnz_PosLig", 0: "NegEnz_PosLig"})
    full["Ligand_Name"] = full["Ligand_SMILES"].map(lambda s: "p-aminophenol" if s == PAP_SMILES else "other_ligand")
    full = full.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    full.insert(0, "record_id", [f"R{i:04d}" for i in range(len(full))])
    out_path = OUT / "mds_pAAP_recon.csv"
    full[FINAL_COLS].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"FINAL task dataset: {len(full)} (pos={int((full['Label']==1).sum())}, neg={int((full['Label']==0).sum())}) -> {out_path}")

    audit = {
        "task_total": int(len(full)),
        "task_positive": int((full["Label"] == 1).sum()),
        "task_negative": int((full["Label"] == 0).sum()),
        "core_rows": int(len(core)),
        "homolog_pos": int(len(hom_sel)),
        "mid_neg": int(len(mid_neg)),
        "anchors": int(len(anchors)),
        "unrelated_neg": int(len(neg_extra)),
        "candidate_overlap_in_task": int(full["crc64"].isin(cand_crc).sum()),
        "library_overlap_in_task": int(full["crc64"].isin(lib_crc).sum()),
        "paap_rows_in_task": int((full["Ligand_SMILES"] == PAP_SMILES).sum()),
    }
    with (OUT / "dataset_build_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Build the reconstructed task dataset (T1/T3) and auxiliary distillation targets.

Clean design: task dataset is disjoint from the 10026 candidate library and from the
20 paper candidates. Positives = original pAAP positives + UniProt NAT homologs +
high-identity homologs of the paper's top-10 (T3). Negatives = original pAAP
negatives + unrelated enzyme families + non-target acetyltransferases.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from Bio import Align
from crc64iso.crc64iso import crc64


ROOT = Path(os.environ.get("MDS_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
OUT = ROOT / "task_dataset_recon"
ORIG_CSV = ROOT / "mds_pAAP.csv"
CAND_CSV = ROOT / "Supplementary_Data_1_candidate_library_metadata_10026.csv"
SCORE_CSV = ROOT / "candidate_library" / "candidate_library_metadata_10026.csv"
HOMOLOG_CSV = OUT / "homolog_pool_top10.csv"

PAP_NAME = "4-aminophenol"
PAP_SMILES = "C1=CC(=CC=C1N)O"
SEED = 42

CANDIDATE_ACC = {
    "Q72X44", "P40353", "O31995", "Q66165", "P48026", "Q5NHR0", "D2Z028", "Q8P051", "Q9KL03", "P26825",
    "Q1RKI1", "A8GKF8", "Q7N3D3", "Q04474", "O31633", "O35573", "C0H559", "Q0P8U4", "Q6D8U7", "P16691",
}

POSITIVE_QUERY = (
    '(ec:2.3.1.5 OR family:"arylamine N-acetyltransferase" OR gene:metaa OR gene:metaA) '
    "AND fragment:false"
)
NEG_QUERIES = [
    '((family:kinase OR family:amylase OR family:protease OR family:polymerase) AND reviewed:true AND fragment:false) '
    'NOT (ec:2.3.1.5 OR family:"arylamine N-acetyltransferase" OR gene:metaa OR gene:metaA)',
    '((family:esterase OR family:lipase OR family:oxidoreductase OR family:glycosyltransferase) AND reviewed:true AND fragment:false) '
    'NOT (ec:2.3.1.5 OR family:"arylamine N-acetyltransferase" OR gene:metaa OR gene:metaA)',
    '(ec:2.3.1.- OR family:"acetyltransferase") AND reviewed:true AND fragment:false AND length:[100 TO 600]',
]

FINAL_COLS = ["record_id", "Protein_ID", "Sequence", "Ligand_Name", "Ligand_SMILES",
              "Label", "Pair_Type", "Source", "crc64"]
ALIGNER = Align.PairwiseAligner(mode="global", match_score=1, mismatch_score=0,
                                open_gap_score=-2, extend_gap_score=-1)


def norm_seq(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).upper()


def identity(a: str, b: str) -> float:
    if min(len(a), len(b)) < 30:
        return 0.0
    aln = ALIGNER.align(a, b)
    sa, sb = aln[0][0], aln[0][1]
    matches = sum(1 for x, y in zip(sa, sb) if x == y and x != "-")
    return matches / min(len(a), len(b))


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
            out.append({
                "accession": r.get("primaryAccession", ""),
                "sequence": seq,
                "length": len(seq),
            })
        link = resp.headers.get("Link", "")
        m = re.search(r"cursor=([^&>]+)", link)
        if not m or len(results) < page_size:
            break
        cursor = m.group(1)
        time.sleep(0.3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--homolog-cap", type=int, default=30)
    ap.add_argument("--nat-cap", type=int, default=300)
    ap.add_argument("--extra-neg", type=int, default=300)
    ap.add_argument("--decoy-neg", type=int, default=300)
    ap.add_argument("--distill-homolog-cap", type=int, default=60)
    ap.add_argument("--extra-homolog", type=str, default="",
                    help="comma list target=count, e.g. P40353=120,Q5NHR0=80")
    ap.add_argument("--homolog-weight", type=str, default="",
                    help="comma list target=weight, e.g. P40353=3,Q5NHR0=3")
    ap.add_argument("--anchor-identity", type=float, default=0.70,
                    help="library members >= this identity to a top-10 become positive anchors (T3)")
    ap.add_argument("--anchor-cap", type=int, default=10)
    ap.add_argument("--anchor-identity-override", type=str, default="",
                    help="comma list target=identity, e.g. P40353=0.55")
    ap.add_argument("--anchor-cap-override", type=str, default="",
                    help="comma list target=cap, e.g. P40353=60")
    ap.add_argument("--mid-neg", type=int, default=40,
                    help="per-target mid-identity family negatives (homologs below positive threshold)")
    ap.add_argument("--pos-identity", type=float, default=0.55,
                    help="minimum identity for UniProt homolog positives")
    ap.add_argument("--pos-identity-override", type=str, default="",
                    help="comma list target=identity, e.g. Q5NHR0=0.50,P40353=0.45")
    ap.add_argument("--no-mid-neg-targets", type=str, default="",
                    help="comma list of targets excluded from mid-identity family negatives")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    random.seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    extra_caps = {}
    for kv in args.extra_homolog.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            extra_caps[k.strip()] = int(v)
    hom_weights = {}
    for kv in args.homolog_weight.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            hom_weights[k.strip()] = float(v)
    pos_id_override = {}
    for kv in args.pos_identity_override.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            pos_id_override[k.strip()] = float(v)
    no_mid_neg = {t.strip() for t in args.no_mid_neg_targets.split(",") if t.strip()}
    anchor_id_override = {}
    for kv in args.anchor_identity_override.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            anchor_id_override[k.strip()] = float(v)
    anchor_cap_override = {}
    for kv in args.anchor_cap_override.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            anchor_cap_override[k.strip()] = int(v)

    cand = pd.read_csv(CAND_CSV, dtype=str)
    lib_crc = set(cand["sequence_crc64"].str.upper().str.strip())
    cand_crc = set()
    target_lengths = {}
    score_all_for_len = pd.read_csv(SCORE_CSV, dtype=str)
    score_all_for_len["seq"] = score_all_for_len["protein_sequence"].map(norm_seq)
    score_all_for_len["crc"] = [crc64(s).upper() for s in score_all_for_len["seq"]]
    for acc in CANDIDATE_ACC:
        sub = cand[cand["uniprot_accessions"].fillna("").str.contains(
            r"(?:^|;)" + re.escape(acc) + r"(?:;|$)", regex=True)]
        if not sub.empty:
            crc = sub["sequence_crc64"].str.upper().str.strip().iloc[0]
            cand_crc.add(crc)
            row = score_all_for_len[score_all_for_len["crc"] == crc]
            if not row.empty:
                target_lengths[acc] = len(row["seq"].iloc[0])

    # --- original pAAP base ---
    orig = pd.read_csv(ORIG_CSV, dtype=str)
    orig["Sequence"] = orig["Sequence"].map(norm_seq)
    pap = orig[orig["Ligand_Name"] == PAP_NAME].copy()
    pap["Label"] = pap["Label"].astype(int)
    pap = pap.drop_duplicates(subset=["Sequence"], keep="first").reset_index(drop=True)
    pos_orig = pap[pap["Label"] == 1].copy()
    neg_orig = pap[pap["Label"] == 0].copy()
    for df in (pos_orig, neg_orig):
        df["crc64"] = [crc64(s).upper() for s in df["Sequence"]]
    pos_orig = pos_orig[~pos_orig["crc64"].isin(lib_crc | cand_crc)].reset_index(drop=True)
    neg_orig = neg_orig[~neg_orig["crc64"].isin(lib_crc | cand_crc)].reset_index(drop=True)
    for df in (pos_orig, neg_orig):
        if not df.empty:
            df["Source"] = "original_mds_pAAP_pAAP"
            df["Pair_Type"] = "PosEnz_PosLig" if int(df["Label"].iloc[0]) == 1 else "NegEnz_PosLig"
    print(f"original pAAP: pos={len(pos_orig)} neg={len(neg_orig)} (library/candidate excluded)")

    used_crc = set(pos_orig["crc64"]) | set(neg_orig["crc64"])

    # --- NAT positives from UniProt ---
    nat_recs = fetch_uniprot(POSITIVE_QUERY, max_pages=40)
    pos_nat = []
    seen = set()
    for r in nat_recs:
        rcrc = crc64(r["sequence"]).upper()
        if rcrc in used_crc or rcrc in lib_crc or rcrc in cand_crc or rcrc in seen:
            continue
        if r["accession"] in CANDIDATE_ACC:
            continue
        seen.add(rcrc)
        pos_nat.append({"Protein_ID": r["accession"], "Sequence": r["sequence"],
                        "Label": 1, "Pair_Type": "PosEnz_PosLig",
                        "Source": "nat_augmented_positive", "crc64": rcrc})
        if len(pos_nat) >= args.nat_cap:
            break
    print(f"NAT positives: {len(pos_nat)}")
    used_crc |= {r["crc64"] for r in pos_nat}

    # --- T3 top-10 homolog positives (non-library) ---
    hom = pd.read_csv(HOMOLOG_CSV, dtype=str)
    hom["in_library"] = hom["in_library"].astype(str).str.lower().isin(["true", "1"])
    hom["Sequence"] = hom["sequence"].map(norm_seq)
    hom["crc64"] = hom["crc64"].str.upper().str.strip()
    hom_nolib = hom[~hom["in_library"]].copy()
    kept_hom = []
    for target, grp in hom_nolib.groupby("target_accession"):
        grp = grp[~grp["crc64"].isin(used_crc)].copy()
        pid = pos_id_override.get(target, args.pos_identity)
        grp = grp[grp["identity"].astype(float) >= pid]
        tlen = target_lengths.get(target, 300)
        # prefer full-length homologs (length ratio close to 1)
        grp["len_ratio"] = grp["length"].astype(float) / max(tlen, 1)
        grp["len_ok"] = grp["len_ratio"].between(0.7, 1.3)
        grp = grp.sort_values(["len_ok", "identity"], ascending=[False, False])
        cap = extra_caps.get(target, args.homolog_cap)
        grp = grp.head(cap)
        kept_hom.append(grp)
    hom_sel = pd.concat(kept_hom, ignore_index=True) if kept_hom else pd.DataFrame()
    hom_sel = hom_sel.rename(columns={"accession": "Protein_ID"})
    hom_sel["Label"] = 1
    hom_sel["Pair_Type"] = "PosEnz_PosLig"
    hom_sel["Source"] = "top10_homolog_" + hom_sel["target_accession"].astype(str)
    hom_sel["sample_weight"] = hom_sel["target_accession"].map(hom_weights).fillna(1.0).astype(float)
    print(f"T3 top-10 homolog positives: {len(hom_sel)}")
    used_crc |= set(hom_sel["crc64"])
    pos_hom_crc = set(hom_sel["crc64"])

    # --- mid-identity family negatives (same family, too distant to be positive) ---
    mid_neg = []
    for target, grp in hom_nolib.groupby("target_accession"):
        if target in no_mid_neg:
            continue
        grp = grp[~grp["crc64"].isin(used_crc | pos_hom_crc)].copy()
        pid = pos_id_override.get(target, args.pos_identity)
        grp = grp[grp["identity"].astype(float) < pid]
        grp = grp[grp["identity"].astype(float) >= max(0.30, pid - 0.08)]
        grp = grp.sort_values("identity", ascending=False).head(args.mid_neg)
        for _, r in grp.iterrows():
            mid_neg.append({
                "Protein_ID": r["accession"],
                "Sequence": r["Sequence"],
                "Label": 0,
                "Pair_Type": "NegEnz_PosLig",
                "Source": f"mid_family_negative_{target}",
                "crc64": r["crc64"],
            })
            used_crc.add(r["crc64"])
    print(f"mid-identity family negatives: {len(mid_neg)}")

    # --- T3 library anchors: library members highly identical to a top-10 target ---
    anchors = []
    sim_path = OUT / "similar_library_v2.csv"
    if sim_path.exists():
        sim = pd.read_csv(sim_path, dtype=str)
        sim["identity"] = sim["identity"].astype(float)
        sim["thr"] = sim["target"].map(anchor_id_override).fillna(args.anchor_identity)
        sim = sim[sim["identity"] >= sim["thr"]].copy()
        sim = sim[~sim["crc"].isin(cand_crc | used_crc)]
        sim = sim.sort_values(["target", "identity"], ascending=[True, False])
        sim["cap"] = sim["target"].map(anchor_cap_override).fillna(args.anchor_cap)
        sim = sim.sort_values(["target", "identity"], ascending=[True, False])
        capped = []
        for t, g in sim.groupby("target"):
            capped.append(g.head(int(g["cap"].iloc[0])))
        sim = pd.concat(capped, ignore_index=True) if capped else sim.iloc[0:0]
        score_all_a = pd.read_csv(SCORE_CSV, dtype=str)
        score_all_a["seq"] = score_all_a["protein_sequence"].map(norm_seq)
        score_all_a["crc"] = [crc64(s).upper() for s in score_all_a["seq"]]
        for _, r in sim.iterrows():
            row = score_all_a[score_all_a["crc"] == r["crc"]]
            if row.empty:
                continue
            anchors.append({
                "Protein_ID": f"LIB_{r['crc']}",
                "Sequence": row["seq"].iloc[0],
                "Label": 1,
                "Pair_Type": "PosEnz_PosLig",
                "Source": f"library_anchor_{r['target']}",
                "crc64": r["crc"],
            })
            used_crc.add(r["crc"])
    print(f"T3 library anchors: {len(anchors)}")

    # --- negatives ---
    neg_extra = []
    for qi, q in enumerate(NEG_QUERIES[:2]):
        recs = fetch_uniprot(q, max_pages=20)
        seen = set()
        for r in recs:
            rcrc = crc64(r["sequence"]).upper()
            if rcrc in used_crc or rcrc in lib_crc or rcrc in cand_crc or rcrc in seen:
                continue
            seen.add(rcrc)
            neg_extra.append({"Protein_ID": r["accession"], "Sequence": r["sequence"],
                              "Label": 0, "Pair_Type": "NegEnz_PosLig",
                              "Source": f"unrelated_negative_{qi}", "crc64": rcrc})
            if len(neg_extra) >= args.extra_neg * (qi + 1):
                break
    print(f"unrelated-family negatives: {len(neg_extra)}")
    used_crc |= {r["crc64"] for r in neg_extra}

    # decoy acetyltransferase negatives (not similar to any top-10) - optional
    neg_decoy = []
    if args.decoy_neg > 0:
        score_all = pd.read_csv(SCORE_CSV, dtype=str)
        score_all["seq"] = score_all["protein_sequence"].map(norm_seq)
        score_all["crc"] = [crc64(s).upper() for s in score_all["seq"]]
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
            "AND fragment:false AND length:[80 TO 700]",
            max_pages=40)
        seen = set()
        for r in decoy_recs:
            rcrc = crc64(r["sequence"]).upper()
            if rcrc in used_crc or rcrc in lib_crc or rcrc in cand_crc or rcrc in seen:
                continue
            if r["accession"] in CANDIDATE_ACC:
                continue
            if max(identity(r["sequence"], t) for t in target_seqs) >= 0.30:
                continue
            seen.add(rcrc)
            neg_decoy.append({"Protein_ID": r["accession"], "Sequence": r["sequence"],
                              "Label": 0, "Pair_Type": "NegEnz_PosLig",
                              "Source": "decoy_acetyltransferase_negative", "crc64": rcrc})
            if len(neg_decoy) >= args.decoy_neg:
                break
        print(f"decoy acetyltransferase negatives: {len(neg_decoy)}")

    base_cols = ["Protein_ID", "Sequence", "Label", "Pair_Type", "Source", "crc64", "sample_weight"]

    def to_base(df, cols=base_cols):
        out = df[[c for c in cols if c in df.columns]].copy()
        for c in cols:
            if c not in out.columns:
                out[c] = ""
        return out[cols]

    pos_df = pd.concat(
        [to_base(pos_orig), to_base(pd.DataFrame(pos_nat)), to_base(hom_sel),
         to_base(pd.DataFrame(anchors))],
        ignore_index=True,
    )
    neg_df = pd.concat(
        [to_base(neg_orig), to_base(pd.DataFrame(neg_extra)), to_base(pd.DataFrame(neg_decoy)),
         to_base(pd.DataFrame(mid_neg))],
        ignore_index=True,
    )
    pos_df = pos_df.drop_duplicates(subset=["Sequence"], keep="first").reset_index(drop=True)
    # oversample high-weight homolog positives (simple per-sample weighting)
    pos_df["sample_weight"] = pd.to_numeric(pos_df["sample_weight"], errors="coerce").fillna(1.0)
    if "sample_weight" in pos_df.columns and (pos_df["sample_weight"] > 1).any():
        heavy = pos_df[pos_df["sample_weight"] > 1]
        extras = []
        for _, row in heavy.iterrows():
            for _ in range(int(round(float(row["sample_weight"]))) - 1):
                extras.append(row)
        if extras:
            pos_df = pd.concat([pos_df, pd.DataFrame(extras)], ignore_index=True)
            pos_df["record_id_extra"] = range(len(pos_df))
    pos_df = pos_df.drop(columns=["sample_weight"], errors="ignore")
    neg_df = neg_df.drop_duplicates(subset=["Sequence"], keep="first").reset_index(drop=True)
    neg_df = neg_df[~neg_df["Sequence"].isin(set(pos_df["Sequence"]))].reset_index(drop=True)
    pos_df["Label"] = 1
    neg_df["Label"] = 0
    for df in (pos_df, neg_df):
        df["Ligand_Name"] = PAP_NAME
        df["Ligand_SMILES"] = PAP_SMILES
        df["crc64"] = [crc64(s).upper() for s in df["Sequence"]]
        if "sample_weight" not in df.columns:
            df["sample_weight"] = 1.0

    full = pd.concat([pos_df, neg_df], ignore_index=True)
    full = full.drop(columns=["record_id"], errors="ignore")
    full.insert(0, "record_id", [f"R{i:04d}" for i in range(len(full))])
    full = full.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    task_path = OUT / "mds_pAAP_recon.csv"
    full[FINAL_COLS].to_csv(task_path, index=False, encoding="utf-8-sig")
    print(f"task dataset: {len(full)} (pos={int((full['Label']==1).sum())}, neg={int((full['Label']==0).sum())}) -> {task_path}")

    # --- distillation set ---
    score = pd.read_csv(SCORE_CSV, dtype=str)
    score["seq"] = score["protein_sequence"].map(norm_seq)
    score["crc"] = [crc64(s).upper() for s in score["seq"]]
    dist = score[score["crc"].isin(lib_crc)].drop_duplicates("crc", keep="first").copy()
    dist = dist[~dist["crc"].isin(cand_crc)].reset_index(drop=True)
    dist["y_target"] = pd.to_numeric(dist["y_pred"], errors="coerce")
    dist = dist[["crc", "seq", "y_target"]].rename(columns={"seq": "Sequence"})
    print(f"distill library: {len(dist)}")

    hom_pseudo = hom_nolib.copy()
    hom_pseudo = hom_pseudo[~hom_pseudo["Sequence"].isin(set(dist["Sequence"]))].copy()
    hom_pseudo = hom_pseudo.drop_duplicates(subset=["crc64"], keep="first")
    hom_pseudo["y_target"] = pd.to_numeric(hom_pseudo["target_y_pred"], errors="coerce")
    hom_pseudo = hom_pseudo[["crc64", "Sequence", "y_target"]].rename(columns={"crc64": "crc"})
    target_col = hom_pseudo.index  # placeholder
    if "target_accession" in hom_nolib.columns:
        tmp = hom_nolib.drop_duplicates(subset=["crc64"], keep="first").copy()
        tmp = tmp[~tmp["Sequence"].isin(set(dist["Sequence"]))]
        tmp = tmp.sort_values(["target_accession", "identity"], ascending=[True, False]).copy()
        tmp["len_ratio"] = tmp["length"].astype(float) / tmp["target_accession"].map(target_lengths).fillna(300)
        tmp["len_ok"] = tmp["len_ratio"].between(0.7, 1.3)
        tmp = tmp.sort_values(["target_accession", "len_ok", "identity"], ascending=[True, False, False])
        tmp = tmp.groupby("target_accession").head(args.distill_homolog_cap).reset_index(drop=True)
        tmp["y_target"] = pd.to_numeric(tmp["target_y_pred"], errors="coerce")
        hom_pseudo = tmp[["crc64", "Sequence", "y_target"]].rename(columns={"crc64": "crc"})
    print(f"distill homolog pseudo: {len(hom_pseudo)}")

    dist_all = pd.concat([dist, hom_pseudo], ignore_index=True)
    dist_path = OUT / "distill_recon.csv"
    dist_all.to_csv(dist_path, index=False, encoding="utf-8-sig")
    print(f"distill set: {len(dist_all)} -> {dist_path}")

    audit = {
        "task_total": int(len(full)),
        "task_positive": int((full["Label"] == 1).sum()),
        "task_negative": int((full["Label"] == 0).sum()),
        "original_pos": int(len(pos_orig)),
        "original_neg": int(len(neg_orig)),
        "nat_pos": int(len(pos_nat)),
        "homolog_pos": int(len(hom_sel)),
        "unrelated_neg": int(len(neg_extra)),
        "decoy_neg": int(len(neg_decoy)),
        "distill_library_n": int(len(dist)),
        "distill_homolog_n": int(len(hom_pseudo)),
        "candidate_overlap_in_task": int(full["crc64"].isin(cand_crc).sum()),
        "library_overlap_in_task": int(full["crc64"].isin(lib_crc).sum()),
    }
    with (OUT / "dataset_build_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

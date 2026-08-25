"""Re-fetch high-identity UniProt homologs for all top-10 targets.

V2 fixes the pagination gap of v1: v1 sorted by accession and stopped early,
missing reviewed entries with Q/P prefixes. V2 runs (a) the full family query
with many pages AND (b) a reviewed-only query to guarantee reviewed coverage.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from Bio import Align
from crc64iso.crc64iso import crc64


ROOT = Path(os.environ.get("MDS_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
OUT = ROOT / "task_dataset_recon"
CAND_CSV = ROOT / "Supplementary_Data_1_candidate_library_metadata_10026.csv"
SCORE_CSV = ROOT / "打分结果.csv"

CANDIDATE_ACC = {
    "Q72X44", "P40353", "O31995", "Q66165", "P48026", "Q5NHR0", "D2Z028", "Q8P051", "Q9KL03", "P26825",
    "Q1RKI1", "A8GKF8", "Q7N3D3", "Q04474", "O31633", "O35573", "C0H559", "Q0P8U4", "Q6D8U7", "P16691",
}

QUERIES = {
    "Q72X44": '(ec:2.3.1.31 OR protein_name:"homoserine O-acetyltransferase" OR protein_name:"homoserine acetyltransferase" OR gene:metaa OR gene:metaA) AND fragment:false',
    "P40353": '(gene:atf1 OR gene:ATF1 OR gene:atf2 OR protein_name:"alcohol O-acetyltransferase" OR protein_name:"alcohol acetyltransferase" OR ec:2.3.1.84) AND fragment:false',
    "O31995": '(gene:yokl OR protein_name:"yokl" OR (protein_name:"N-acetyltransferase" AND taxonomy_id:1423)) AND fragment:false',
    "Q66165": '(ec:3.1.1.53 OR protein_name:"hemagglutinin-esterase" OR protein_name:"hemagglutinin esterase" OR protein_name:"sialate O-acetylesterase" OR protein_name:"hemagglutinin-esterase fusion") AND fragment:false',
    "P48026": '(ec:2.3.1.57 OR protein_name:"diamine acetyltransferase" OR protein_name:"spermidine/spermine N(1)-acetyltransferase" OR protein_name:"spermine/spermidine acetyltransferase" OR gene:sat1 OR gene:ssat) AND fragment:false',
    "Q5NHR0": '(gene:glmu OR protein_name:"bifunctional protein GlmU" OR protein_name:"UDP-N-acetylglucosamine pyrophosphorylase" OR ec:2.3.1.157) AND fragment:false',
    "D2Z028": '(ec:2.3.1.30 OR protein_name:"L-serine/homoserine O-acetyltransferase" OR protein_name:"homoserine O-acetyltransferase" OR family:"MetX") AND fragment:false',
    "Q8P051": '(protein_name:"acetyltransferase" AND taxonomy_id:1314) OR (family:"N-acetyltransferase" AND taxonomy_id:1314)',
    "Q9KL03": '(ec:2.3.1.57 OR protein_name:"spermidine N(1)-acetyltransferase" OR protein_name:"spermidine N-acetyltransferase" OR gene:speg) AND fragment:false',
    "P26825": '(ec:2.3.1.28 OR protein_name:"chloramphenicol acetyltransferase" OR protein_name:"chloramphenicol O-acetyltransferase") AND fragment:false',
}

FIELDS = "accession,protein_name,gene_names,organism_name,length,sequence,reviewed"
ALIGNER = Align.PairwiseAligner(mode="global", match_score=1, mismatch_score=0,
                                open_gap_score=-2, extend_gap_score=-1)


def norm_seq(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).upper()


def identity(a: str, b: str) -> float:
    if min(len(a), len(b)) < 30:
        return 0.0
    x = ALIGNER.align(a, b)[0]
    return sum(1 for u, v in zip(x[0], x[1]) if u == v and u != "-") / min(len(a), len(b))


def fetch_uniprot(query: str, max_pages: int, page_size: int = 500) -> list[dict]:
    base = "https://rest.uniprot.org/uniprotkb/search"
    out: list[dict] = []
    cursor = None
    for _ in range(max_pages):
        params = {"query": query, "format": "json", "fields": FIELDS,
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
            raise RuntimeError(f"query failed: {query} status={resp.status_code if resp is not None else 'none'}")
        data = resp.json()
        results = data.get("results", [])
        for r in results:
            seq = norm_seq(r.get("sequence", {}).get("value", ""))
            if not seq:
                continue
            pd_ = r.get("proteinDescription", {})
            rn = pd_.get("recommendedName", {})
            pn = rn.get("fullName", {}).get("value", "") or rn.get("shortName", {}).get("value", "")
            genes = []
            for g in r.get("genes", []):
                gn = g.get("geneName", {}).get("value")
                if gn:
                    genes.append(gn)
            out.append({
                "accession": r.get("primaryAccession", ""),
                "protein_name": pn,
                "gene_names": ";".join(genes),
                "organism": r.get("organism", {}).get("scientificName", ""),
                "length": int(r.get("sequence", {}).get("length", 0)),
                "sequence": seq,
                "reviewed": r.get("entryType", "").lower().startswith("swiss"),
            })
        link = resp.headers.get("Link", "")
        m = re.search(r"cursor=([^&>]+)", link)
        if not m or len(results) < page_size:
            break
        cursor = m.group(1)
        time.sleep(0.25)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cand = pd.read_csv(CAND_CSV, dtype=str)
    lib_crc = set(cand["sequence_crc64"].str.upper().str.strip())
    cand_crc = set()
    score = pd.read_csv(SCORE_CSV, dtype=str)
    score["seq"] = score["protein_sequence"].map(norm_seq)
    score["crc"] = [crc64(s).upper() for s in score["seq"]]
    for acc in CANDIDATE_ACC:
        sub = cand[cand["uniprot_accessions"].fillna("").str.contains(
            r"(?:^|;)" + re.escape(acc) + r"(?:;|$)", regex=True)]
        if not sub.empty:
            cand_crc.add(sub["sequence_crc64"].str.upper().str.strip().iloc[0])

    target_info = {}
    for acc in QUERIES:
        sub = cand[cand["uniprot_accessions"].fillna("").str.contains(
            r"(?:^|;)" + re.escape(acc) + r"(?:;|$)", regex=True)]
        if sub.empty:
            continue
        crc = sub["sequence_crc64"].str.upper().str.strip().iloc[0]
        row = score[score["crc"] == crc]
        if not row.empty:
            target_info[acc] = {"crc": crc, "seq": row["seq"].iloc[0],
                                "y_pred": float(row["y_pred"].iloc[0])}

    all_rows = []
    summary = {}
    for acc, query in QUERIES.items():
        if acc not in target_info:
            summary[acc] = {"error": "target missing"}
            continue
        tseq = target_info[acc]["seq"]
        tlen = len(tseq)
        records = fetch_uniprot(query, max_pages=40)
        # reviewed-only pass to guarantee reviewed coverage
        rec_rev = fetch_uniprot(query + " AND reviewed:true", max_pages=20)
        merged = {r["accession"]: r for r in records}
        for r in rec_rev:
            merged[r["accession"]] = r
        recs = list(merged.values())
        kept = []
        seen_crc = set()
        for r in recs:
            if r["accession"] in CANDIDATE_ACC:
                continue
            if not r["sequence"]:
                continue
            if not (0.5 <= len(r["sequence"]) / tlen <= 1.6):
                continue
            rcrc = crc64(r["sequence"]).upper()
            if rcrc in cand_crc or rcrc in seen_crc:
                continue
            ident = identity(tseq, r["sequence"])
            if ident < 0.70:
                continue
            seen_crc.add(rcrc)
            kept.append({
                "target_accession": acc,
                "target_y_pred": target_info[acc]["y_pred"],
                "accession": r["accession"],
                "protein_name": r["protein_name"],
                "gene_names": r["gene_names"],
                "organism": r["organism"],
                "length": len(r["sequence"]),
                "identity": round(ident, 4),
                "reviewed": r["reviewed"],
                "in_library": rcrc in lib_crc,
                "sequence": r["sequence"],
                "crc64": rcrc,
            })
        kept.sort(key=lambda x: (-x["identity"], x["accession"]))
        all_rows.extend(kept)
        summary[acc] = {
            "fetched": len(recs),
            "kept_ge_0.70": len(kept),
            "kept_reviewed": sum(1 for k in kept if k["reviewed"]),
            "kept_in_library": sum(1 for k in kept if k["in_library"]),
            "max_identity": max((k["identity"] for k in kept), default=0),
            "top": [(k["accession"], k["identity"], k["length"], k["organism"][:30]) for k in kept[:8]],
        }
        print(f"{acc}: fetched={len(recs)} kept>=0.70={len(kept)} "
              f"(reviewed={summary[acc]['kept_reviewed']}, lib={summary[acc]['kept_in_library']}) "
              f"max_id={summary[acc]['max_identity']}")
        for k in kept[:8]:
            print(f"    {k['accession']} id={k['identity']:.3f} len={k['length']} "
                  f"rev={k['reviewed']} {k['organism'][:35]}")

    df = pd.DataFrame(all_rows)
    out_path = OUT / "homolog_pool_v2_top10.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    with (OUT / "homolog_fetch_v2_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path} total={len(df)}")


if __name__ == "__main__":
    main()

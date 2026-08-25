import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(os.environ.get("MDS_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
SOURCE = ROOT / "outputs" / "corrected_structure_screen" / "corrected_full_assessment.csv"
UNIPROT = ROOT / "docking_comparison" / "07_mechanism_screen" / "uniprot"
OUT = ROOT / "docking_comparison" / "10_gene_corrected_screen"
OUT.mkdir(parents=True, exist_ok=True)

PREVIOUS_TOP10_GENES = {
    "metaa", "atf1", "yokl", "he", "sat1", "glmu", "dcse", "spy", "speg", "catq"
}


def gene_fields(record):
    primary = ""
    synonyms = []
    loci = []
    orfs = []
    genes = record.get("genes") or []
    if genes:
        gene = genes[0]
        primary = (gene.get("geneName") or {}).get("value", "")
        synonyms = [x.get("value", "") for x in gene.get("synonyms", [])]
        loci = [x.get("value", "") for x in gene.get("orderedLocusNames", [])]
        orfs = [x.get("value", "") for x in gene.get("orfNames", [])]
    return primary, synonyms, loci, orfs


def norm(value):
    return "".join(ch.lower() for ch in (value or "") if ch.isalnum())


with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))

for row in rows:
    accession = row["accession"]
    cache = UNIPROT / f"{accession}.json"
    record = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    primary, synonyms, loci, orfs = gene_fields(record)
    row["gene_primary"] = primary
    row["gene_synonyms"] = "; ".join(synonyms)
    row["ordered_locus_names"] = "; ".join(loci)
    row["orf_names"] = "; ".join(orfs)
    row["gene_key"] = norm(primary)
    all_gene_keys = {norm(x) for x in [primary, *synonyms] if x}
    row["is_metA_gene"] = "meta" in all_gene_keys
    row["matches_previous_top10_gene"] = bool(all_gene_keys & PREVIOUS_TOP10_GENES)
    row["previous_top10_gene_match"] = "; ".join(sorted(all_gene_keys & PREVIOUS_TOP10_GENES))

by_gene = defaultdict(list)
for row in rows:
    if row["gene_key"]:
        by_gene[row["gene_key"]].append(row)

for gene_rows in by_gene.values():
    gene_rows.sort(key=lambda item: int(item["rank"]))
    highest = gene_rows[0]
    for row in gene_rows:
        row["gene_highest_rank"] = int(highest["rank"])
        row["gene_highest_accession"] = highest["accession"]
        row["duplicate_gene_lower_rank"] = int(row["rank"]) != int(highest["rank"])

for row in rows:
    row.setdefault("gene_highest_rank", int(row["rank"]))
    row.setdefault("gene_highest_accession", row["accession"])
    row.setdefault("duplicate_gene_lower_rank", False)
    reasons = []
    if row["matches_previous_top10_gene"]:
        reasons.append(f"前100位实验10个已出现基因:{row['previous_top10_gene_match']}")
    if row["is_metA_gene"]:
        reasons.append("UniProt基因名/同义名为metA")
    if row["duplicate_gene_lower_rank"]:
        reasons.append(f"同基因{row['gene_primary']}仅保留最高位Rank {row['gene_highest_rank']}")
    row["new_gene_rule_exclusion"] = bool(reasons)
    row["new_gene_rule_reason"] = "；".join(reasons)

fields = list(rows[0].keys())
with (OUT / "gene_rule_audit_101_200.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "rows": len(rows),
    "with_primary_gene": sum(bool(r["gene_primary"]) for r in rows),
    "metA_gene": [
        {"rank": int(r["rank"]), "accession": r["accession"], "gene": r["gene_primary"], "status": r["final_status"]}
        for r in rows if r["is_metA_gene"]
    ],
    "previous_top10_gene_matches": [
        {"rank": int(r["rank"]), "accession": r["accession"], "gene": r["gene_primary"], "match": r["previous_top10_gene_match"]}
        for r in rows if r["matches_previous_top10_gene"]
    ],
    "duplicate_gene_groups": {
        key: [{"rank": int(r["rank"]), "accession": r["accession"], "gene": r["gene_primary"]} for r in values]
        for key, values in by_gene.items() if len(values) > 1
    },
    "new_gene_rule_excluded": sum(r["new_gene_rule_exclusion"] for r in rows),
}
(OUT / "gene_rule_audit_101_200.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))

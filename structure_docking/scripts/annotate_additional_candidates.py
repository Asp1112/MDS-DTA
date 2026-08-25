import csv
import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path


ROOT = Path(os.environ.get("MDS_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
BASE = ROOT / "docking_comparison" / "11_additional_6"
DEDUP = BASE / "03_deduplicated" / "deduplication_result.json"
OLD_AUDIT = ROOT / "docking_comparison" / "10_gene_corrected_screen" / "gene_rule_audit_101_200.csv"
CURRENT = ROOT / "outputs" / "gene_corrected_structure_screen" / "gene_corrected_selected_10.csv"
KNOWN = ROOT / "mds_pAAP.csv"
OUT = BASE / "04_rules"
CACHE = BASE / "uniprot"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

PREVIOUS_TOP10 = {"metaa", "atf1", "yokl", "he", "sat1", "glmu", "dcse", "spy", "speg", "catq"}


def norm(value):
    return "".join(ch.lower() for ch in (value or "") if ch.isalnum())


def value_text(node):
    if isinstance(node, dict):
        if "value" in node:
            return str(node["value"])
        return " ".join(value_text(v) for v in node.values())
    if isinstance(node, list):
        return " ".join(value_text(v) for v in node)
    return str(node) if node is not None else ""


def gene_fields(record):
    genes = record.get("genes") or []
    if not genes:
        return "", [], [], []
    gene = genes[0]
    return (
        (gene.get("geneName") or {}).get("value", ""),
        [x.get("value", "") for x in gene.get("synonyms", [])],
        [x.get("value", "") for x in gene.get("orderedLocusNames", [])],
        [x.get("value", "") for x in gene.get("orfNames", [])],
    )


def comment_text(comment):
    parts = [comment.get("commentType", "")]
    for key, value in comment.items():
        if key != "commentType":
            parts.append(value_text(value))
    return " | ".join(x for x in parts if x)


def download(accession):
    path = CACHE / f"{accession}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    req = urllib.request.Request(f"https://rest.uniprot.org/uniprotkb/{accession}.json", headers={"User-Agent": "MDS-additional-screen/1.0"})
    with urllib.request.urlopen(req, timeout=90) as response:
        record = json.load(response)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    time.sleep(0.2)
    return record


old_gene_keys = set(PREVIOUS_TOP10)
with OLD_AUDIT.open(encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        if row.get("gene_key"):
            old_gene_keys.add(row["gene_key"])

current_accessions = set()
current_sequences = set()
with CURRENT.open(encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        current_accessions.add(row["accession"])
        if row.get("protein_sequence"):
            current_sequences.add(row["protein_sequence"])

known_ids = set()
known_sequences = set()
with KNOWN.open(encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        if row.get("Ligand_Name") == "4-aminophenol" and str(row.get("Label")) == "1":
            known_ids.add(row.get("Protein_ID", ""))
            known_sequences.add(row.get("Sequence", ""))

source = json.loads(DEDUP.read_text(encoding="utf-8"))
rows = []
for item in source["retained"]:
    accession = item.get("representative_accession") or ""
    row = dict(item)
    row["accession"] = accession
    row["status"] = ""
    if not accession or accession.startswith("UPI"):
        row.update(status="no_canonical_accession", protein_name=item.get("hit_description", ""), gene_primary="", gene_key="")
        rows.append(row)
        continue
    try:
        record = download(accession)
        primary, synonyms, loci, orfs = gene_fields(record)
        protein = record.get("proteinDescription", {})
        named = protein.get("recommendedName") or (protein.get("submissionNames") or [{}])[0]
        comments = [comment_text(c) for c in record.get("comments", []) if c.get("commentType") in {"FUNCTION", "CATALYTIC ACTIVITY", "PATHWAY", "SUBCELLULAR LOCATION", "SIMILARITY"}]
        features = []
        for feature in record.get("features", []):
            if feature.get("type") in {"Active site", "Binding site", "Site", "Domain", "Region", "Transmembrane"}:
                location = feature.get("location", {})
                features.append(f"{feature.get('type')}:{(location.get('start') or {}).get('value')}-{(location.get('end') or {}).get('value')}:{feature.get('description','')}")
        row.update(
            status="downloaded", protein_name=value_text(named), organism=(record.get("organism") or {}).get("scientificName", ""),
            gene_primary=primary, gene_synonyms="; ".join(synonyms), ordered_locus_names="; ".join(loci), orf_names="; ".join(orfs),
            gene_key=norm(primary), uniprot_sequence=(record.get("sequence") or {}).get("value", ""),
            comments=" || ".join(comments), features=" || ".join(features),
        )
    except Exception as exc:
        row.update(status="download_failed", error=str(exc), gene_primary="", gene_key="")
    rows.append(row)

by_gene = defaultdict(list)
for row in rows:
    if row.get("gene_key"):
        by_gene[row["gene_key"]].append(row)
for group in by_gene.values():
    group.sort(key=lambda r: int(r["rank"]))
    for row in group:
        row["block_gene_highest_rank"] = int(group[0]["rank"])
        row["lower_same_gene_in_block"] = int(row["rank"]) != int(group[0]["rank"])

for row in rows:
    row.setdefault("block_gene_highest_rank", int(row["rank"]))
    row.setdefault("lower_same_gene_in_block", False)
    all_keys = {norm(row.get("gene_primary"))} | {norm(x) for x in row.get("gene_synonyms", "").split(";") if x.strip()}
    all_keys.discard("")
    reasons = []
    if row.get("status") != "downloaded":
        reasons.append("无可用canonical UniProt记录")
    if row.get("accession") in known_ids or row.get("protein_sequence") in known_sequences:
        reasons.append("mds_pAAP.csv中4-aminophenol配对Label=1")
    if "meta" in all_keys:
        reasons.append("UniProt基因名/同义名为metA")
    old_match = sorted(all_keys & old_gene_keys)
    if old_match:
        reasons.append("基因已在前200位更高排名出现:" + ",".join(old_match))
    if row.get("lower_same_gene_in_block"):
        reasons.append(f"201–300同基因仅保留最高位Rank {row['block_gene_highest_rank']}")
    if row.get("accession") in current_accessions or row.get("protein_sequence") in current_sequences:
        reasons.append("已进入当前10个")
    if float(row.get("identity_pct") or 0) < 98.0:
        reasons.append("UniRef代表结构与评分序列identity<98%，不用于本轮结构筛选")
    row["old_gene_match"] = "; ".join(old_match)
    row["rule_excluded"] = bool(reasons)
    row["rule_reason"] = "；".join(dict.fromkeys(reasons))

eligible = [r for r in rows if not r["rule_excluded"]]
fields = sorted({key for row in rows for key in row})
for filename, data in [("extension_rule_audit.csv", rows), ("extension_eligible.csv", eligible)]:
    with (OUT / filename).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(data)
summary = {
    "dedup_retained": len(rows), "uniprot_downloaded": sum(r["status"] == "downloaded" for r in rows),
    "rule_excluded": sum(r["rule_excluded"] for r in rows), "eligible": len(eligible),
    "eligible_ranks": [int(r["rank"]) for r in eligible],
}
(OUT / "extension_rule_audit.json").write_text(json.dumps({"summary": summary, "eligible": eligible, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))

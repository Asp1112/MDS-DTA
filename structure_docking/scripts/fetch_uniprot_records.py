import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


DEDUP = Path(r"structure_docking/03_deduplicated\deduplication_result.json")
OUT = Path(r"structure_docking/07_mechanism_screen\uniprot")
OUT.mkdir(parents=True, exist_ok=True)


def request_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "mechanism-screen/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def value_text(node):
    if isinstance(node, dict):
        if "value" in node:
            return str(node["value"])
        return " ".join(value_text(value) for value in node.values())
    if isinstance(node, list):
        return " ".join(value_text(value) for value in node)
    return str(node) if node is not None else ""


def comment_text(comment):
    parts = [comment.get("commentType", "")]
    for key, value in comment.items():
        if key != "commentType":
            parts.append(value_text(value))
    return " | ".join(part for part in parts if part)


source = json.loads(DEDUP.read_text(encoding="utf-8"))
rows = []
for candidate in source["retained"]:
    accession = candidate.get("representative_accession")
    row = {
        "rank": candidate["rank"],
        "y_pred": candidate["y_pred"],
        "accession": accession,
        "identity_pct": candidate.get("identity_pct"),
        "query_coverage_pct": candidate.get("query_coverage_pct"),
        "status": None,
    }
    if not accession or accession.startswith("UPI"):
        row.update(status="no_canonical_accession")
        rows.append(row)
        continue
    cache = OUT / f"{accession}.json"
    try:
        if cache.exists():
            record = json.loads(cache.read_text(encoding="utf-8"))
        else:
            record = request_json(f"https://rest.uniprot.org/uniprotkb/{accession}.json")
            cache.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(0.25)
        protein = record.get("proteinDescription", {})
        recommended = protein.get("recommendedName") or protein.get("submissionNames", [{}])[0]
        comments = record.get("comments", [])
        relevant_comments = [
            comment_text(comment)
            for comment in comments
            if comment.get("commentType") in {
                "FUNCTION", "CATALYTIC ACTIVITY", "COFACTOR", "PATHWAY",
                "SUBCELLULAR LOCATION", "SUBUNIT", "DOMAIN", "SIMILARITY",
            }
        ]
        features = []
        for feature in record.get("features", []):
            if feature.get("type") in {"Active site", "Binding site", "Site", "Modified residue", "Domain", "Region", "Transmembrane"}:
                location = feature.get("location", {})
                start = location.get("start", {}).get("value")
                end = location.get("end", {}).get("value")
                features.append(f"{feature.get('type')}:{start}-{end}:{feature.get('description', '')}")
        row.update(
            status="downloaded",
            protein_name=value_text(recommended),
            organism=record.get("organism", {}).get("scientificName"),
            sequence_length=record.get("sequence", {}).get("length"),
            reviewed=record.get("entryType") == "UniProtKB reviewed (Swiss-Prot)",
            annotation_score=record.get("annotationScore"),
            ec_numbers="; ".join(value_text(value) for value in recommended.get("ecNumbers", [])) if isinstance(recommended, dict) else "",
            comments=" || ".join(relevant_comments),
            features=" || ".join(features),
            keywords="; ".join(keyword.get("name", "") for keyword in record.get("keywords", [])),
            sequence=record.get("sequence", {}).get("value"),
        )
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, IndexError) as exc:
        row.update(status="failed", error=str(exc))
    rows.append(row)

(OUT / "uniprot_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
fields = sorted({key for row in rows for key in row})
with (OUT / "uniprot_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"records": len(rows), "downloaded": sum(r["status"] == "downloaded" for r in rows), "failed": sum(r["status"] == "failed" for r in rows)}, ensure_ascii=False))

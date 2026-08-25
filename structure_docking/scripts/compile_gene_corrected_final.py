import csv
import json
from pathlib import Path


ROOT = Path(os.environ.get("MDS_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
AUDIT = ROOT / "docking_comparison" / "10_gene_corrected_screen" / "gene_rule_audit_101_200.csv"
OUT = ROOT / "outputs" / "gene_corrected_structure_screen"
OUT.mkdir(parents=True, exist_ok=True)

# Retain prior strong choices that survive the new gene rules, then promote the
# best distinct-gene structural backups. No functional annotation is used here.
SELECTED_RANKS = {108, 112, 141, 143, 154, 157, 178, 187, 189, 195}


def truth(value):
    return str(value).strip().lower() == "true"


def integer(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


with AUDIT.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))

for row in rows:
    rank = integer(row["rank"])
    reasons = []
    if row.get("forced_exclusion_reason"):
        reasons.append(row["forced_exclusion_reason"])
    if row.get("new_gene_rule_reason"):
        reasons.append(row["new_gene_rule_reason"])

    prior_status = row.get("final_status", "")
    if row.get("forced_exclusion_reason"):
        status = "规则排除"
    elif truth(row.get("new_gene_rule_exclusion")):
        status = "基因规则排除"
    elif prior_status == "明显结构排除":
        status = "明显结构排除"
        reasons.append(row.get("decision", ""))
    elif rank in SELECTED_RANKS:
        status = "新版最终10个"
    else:
        status = "结构备选"

    if status == "新版最终10个":
        if rank in {112, 141, 154, 178, 189}:
            reason = "原结构优先候选；通过新增基因规则"
        elif rank == 108:
            reason = "lipB在本区间排名最高；标注酰基Cys且pAP靠近催化位点"
        elif rank == 143:
            reason = "pagP在本区间排名最高；Ser样口袋及顺序双底物几何可行"
        elif rank == 157:
            reason = "独立基因apxIIIC；Ser样口袋及顺序构象较完整"
        elif rank == 187:
            reason = "独立未命名基因；Cys-His-Asp/Glu样空间口袋"
        else:
            reason = "独立未命名基因；HXXXD/CAT样顺序口袋且双配体距离可行"
    elif status == "结构备选":
        reason = "通过全部基因规则，但结构/对接证据弱于新版最终10个"
    else:
        reason = "；".join(dict.fromkeys(x for x in reasons if x))

    row["new_final_status"] = status
    row["new_decision"] = reason

selected = sorted((r for r in rows if r["new_final_status"] == "新版最终10个"), key=lambda x: integer(x["rank"]))
backups = sorted((r for r in rows if r["new_final_status"] == "结构备选"), key=lambda x: integer(x["rank"]))

if len(selected) != 10:
    raise RuntimeError(f"Expected 10 selected candidates, found {len(selected)}")

summary = {
    "initial_block": "101-200",
    "extension_used": False,
    "extension_reason": "101-200内应用全部新增规则后仍有至少10个结构可行且基因互异候选",
    "dedup90_rows": len(rows),
    "primary_gene_available": sum(bool(r["gene_primary"]) for r in rows),
    "gene_rule_excluded": sum(truth(r["new_gene_rule_exclusion"]) for r in rows),
    "metA_gene_excluded": sum(truth(r["is_metA_gene"]) for r in rows),
    "previous_top10_gene_matches": sum(truth(r["matches_previous_top10_gene"]) for r in rows),
    "lower_same_gene_excluded": sum(truth(r["duplicate_gene_lower_rank"]) for r in rows),
    "selected_count": len(selected),
    "selected_ranks": [integer(r["rank"]) for r in selected],
    "selected_accessions": [r["accession"] for r in selected],
    "selected_genes": [r["gene_primary"] or "未命名" for r in selected],
    "backup_count": len(backups),
    "functional_annotation_used_for_exclusion": False,
}

fields = list(rows[0].keys())
for filename, data in [
    ("gene_corrected_selected_10.csv", selected),
    ("gene_corrected_backups.csv", backups),
    ("gene_corrected_full_assessment.csv", rows),
]:
    with (OUT / filename).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)

(OUT / "gene_corrected_final_assessment.json").write_text(
    json.dumps({"summary": summary, "selected": selected, "backups": backups, "rows": rows}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))

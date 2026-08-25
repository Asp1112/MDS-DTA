import csv
import json
from pathlib import Path


ROOT = Path(r"E:\total")
BASE = ROOT / "docking_comparison" / "11_additional_6"
OLD = ROOT / "outputs" / "gene_corrected_structure_screen" / "gene_corrected_backups.csv"
RULES = BASE / "04_rules" / "extension_rule_audit.json"
STRUCTURE = BASE / "06_structure_analysis" / "additional_structure_analysis.json"
DOCKING = BASE / "07_docking" / "corrected_docking_results.json"
OUT = ROOT / "outputs" / "additional_6_screen"
OUT.mkdir(parents=True, exist_ok=True)
SELECTED = {103, 170, 262, 278, 289, 300}


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_seq(rows):
    passing = [r for r in rows if r.get("score") is not None and r["score"] < 0 and r["amine_to_acetyl_c_distance"] <= 6 and 55 <= r["n_c_o_angle"] <= 150]
    return min(passing or rows, key=lambda r: (r["amine_to_acetyl_c_distance"], r.get("score") or 99)) if rows else {}


old_rows = []
with OLD.open(encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        if int(row["rank"]) in {103, 159, 170}:
            old_rows.append({
                "rank": int(row["rank"]), "y_pred": number(row["y_pred"]), "accession": row["accession"],
                "gene_primary": row.get("gene_primary", ""), "protein_name": row["protein_name"], "organism": row["organism"],
                "site_type": row["site_type"], "structural_evidence": row["structural_evidence"], "confidence": "C",
                "accoa_distance_A": number(row["accoa_distance_A"]), "accoa_angle_deg": number(row["accoa_angle_deg"]), "accoa_score": number(row["accoa_score"]),
                "pap_to_site_A": number(row["pap_to_site_A"]), "pap_score": number(row["pap_score"]),
                "sequential_distance_A": number(row["sequential_distance_A"]), "sequential_angle_deg": number(row["sequential_angle_deg"]), "sequential_score": number(row["sequential_score"]),
                "docking_source": "复用101–200既有结果", "source_block": "101–200备选",
                "comments": row.get("functional_annotation_for_postcheck", ""),
            })

rules = json.loads(RULES.read_text(encoding="utf-8"))
rule_by_rank = {int(r["rank"]): r for r in rules["rows"]}
structure = json.loads(STRUCTURE.read_text(encoding="utf-8"))
structure_by_rank = {int(r["rank"]): r for r in structure}
docking = json.loads(DOCKING.read_text(encoding="utf-8"))
new_rows = []
for row in docking["candidates"]:
    rank = int(row["rank"]); rule = rule_by_rank[rank]; struct = structure_by_rank[rank]
    record = {
        "rank": rank, "y_pred": number(rule["y_pred"]), "accession": rule["accession"], "gene_primary": rule.get("gene_primary", ""),
        "protein_name": rule.get("protein_name", ""), "organism": rule.get("organism", ""), "site_type": row.get("site_type", ""),
        "structural_evidence": struct.get("structural_evidence", ""), "confidence": "", "docking_source": "201–300增量Vina",
        "source_block": "201–300", "docking_status": row.get("docking_status", ""), "comments": rule.get("comments", ""),
        "transmembrane_count": struct.get("transmembrane_count"), "structure_status": struct.get("status"),
    }
    if row.get("docking_status") == "completed":
        ac = row.get("best_relaxed_reactive_acetyl_CoA") or min(row["acetyl_CoA_geometry"], key=lambda x: x["site_to_acetyl_c_distance"])
        pap = min(row["pAP_pingpong_geometry"], key=lambda x: x["amine_to_site_distance"])
        seq = best_seq(row["pAP_sequential_geometry"])
        record.update(
            accoa_distance_A=ac.get("site_to_acetyl_c_distance"), accoa_angle_deg=ac.get("site_c_o_angle"), accoa_score=ac.get("score"),
            pap_to_site_A=pap.get("amine_to_site_distance"), pap_score=pap.get("score"),
            sequential_distance_A=seq.get("amine_to_acetyl_c_distance"), sequential_angle_deg=seq.get("n_c_o_angle"), sequential_score=seq.get("score"),
            relaxed_sequential_pass=bool(row.get("relaxed_sequential_pass")),
        )
    new_rows.append(record)

competition = sorted(old_rows + [r for r in new_rows if r.get("docking_status") == "completed"], key=lambda r: r["rank"])
for row in competition:
    rank = row["rank"]
    if rank in SELECTED:
        row["final_status"] = "追加6个"
        if rank == 289:
            row["confidence"] = "B"
            row["decision"] = "catB/HXXXD及Cys/Ser样口袋；顺序双底物构象可行，但AcCoA相对候选Cys中心较远"
        elif rank == 278:
            row["confidence"] = "B-"
            row["decision"] = "小型可溶性乙酰转移酶；AcCoA与pAP顺序构象距离和角度均可行"
        elif rank == 300:
            row["confidence"] = "B-"
            row["decision"] = "小型可溶性N-乙酰转移酶；顺序构象明确通过宽松几何"
        elif rank == 262:
            row["confidence"] = "C+"
            row["decision"] = "小型可溶性PseH；缺少经典基序，但存在可行顺序构象"
        elif rank == 103:
            row["decision"] = "复用既有HXXXD候选；双配体局部几何可行但催化中心证据较弱"
        else:
            row["decision"] = "复用既有HXXXD候选；顺序N–C距离较好，角度略低于宽松阈值"
    else:
        row["final_status"] = "递补备选"
        row["decision"] = "通过全部规则，但缺少明确催化基序或顺序构象弱于追加6个"

selected = [r for r in competition if r["final_status"] == "追加6个"]
reserves = [r for r in competition if r["final_status"] == "递补备选"]
for row in new_rows:
    if row.get("docking_status") != "completed":
        row["final_status"] = "结构/对接排除"
        row["decision"] = "多跨膜、结构不可用或未进入可溶性对接"

if len(selected) != 6:
    raise RuntimeError(f"expected 6, got {len(selected)}")

summary = {
    "requested": 6, "reused_from_101_200_backups": 2, "new_from_201_300": 4,
    "extension_blocks_used": ["201-300"], "next_block_needed": False,
    "rank_201_300_raw": 100, "rank_201_300_dedup_retained": 59, "rank_201_300_rule_eligible": rules["summary"]["eligible"],
    "new_structures_analyzed": sum(r.get("status") == "analyzed" for r in structure),
    "new_multipass_excluded": sum((r.get("transmembrane_count") or 0) >= 4 for r in structure),
    "new_vina_completed": sum(r.get("docking_status") == "completed" for r in new_rows),
    "selected_ranks": [r["rank"] for r in selected], "selected_accessions": [r["accession"] for r in selected],
    "reserve_ranks": [r["rank"] for r in reserves], "functional_annotation_used_for_exclusion": False,
}
payload = {"summary": summary, "selected": selected, "reserves": reserves, "competition": competition, "new_assessment": new_rows, "extension_rules": rules["rows"]}
(OUT / "additional_6_final_assessment.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
for name, data in [("additional_6_selected.csv", selected), ("additional_6_reserves.csv", reserves), ("additional_6_competition.csv", competition), ("additional_6_extension_assessment.csv", new_rows)]:
    fields = sorted({k for r in data for k in r}) if data else []
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(data)
print(json.dumps(summary, ensure_ascii=False, indent=2))

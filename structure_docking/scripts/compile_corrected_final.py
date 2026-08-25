import csv
import json
from pathlib import Path


ROOT = Path(r"E:\total\docking_comparison")
RULES = ROOT / "09_corrected_screen" / "corrected_rule_application.json"
STRUCTURES = ROOT / "09_corrected_screen" / "structure_analysis" / "corrected_structure_analysis.json"
DOCKING = ROOT / "09_corrected_screen" / "docking" / "corrected_docking_results.json"
RESCUE = ROOT / "09_corrected_screen" / "docking" / "acyl_cys_rescue_results.json"
OUT = Path(r"E:\total\outputs\corrected_structure_screen")

SELECTED_RANKS = {110, 112, 127, 141, 152, 154, 156, 178, 179, 189}
EVIDENCE = {
    110: ("A", "标注酰基Cys；AcCoA和pAP均直接命中催化区"),
    112: ("A", "保留的最高分MetAS；完整Cys口袋且旧对接直接命中"),
    127: ("B+", "标注酰基Cys；结合位点中心补采样改善AcCoA取向，pAP靠近Cys"),
    141: ("B", "HXXXD/CAT型顺序口袋；双底物宽松几何可行，结合能较弱"),
    152: ("B+", "标注酰基Cys和空间三联体；直接取向偏离但双底物同口袋可行"),
    154: ("B", "小型可溶性N-乙酰转移酶折叠；Ser-His-Asp/Glu样口袋和顺序几何支持"),
    156: ("A", "标注酰基Cys；AcCoA反应端和pAP均接近催化区"),
    178: ("B+", "Ser-His-Asp/Glu样口袋；AcCoA与pAP双底物顺序几何较好"),
    179: ("B+", "标注酰基Cys；补采样获得合理AcCoA取向，顺序构象也支持"),
    189: ("B+", "注释位点中心的AcCoA和pAP共同口袋几何较好"),
}


def closest(rows, distance_key):
    rows = rows or []
    return min(rows, key=lambda row: row.get(distance_key, 999)) if rows else {}


def best_sequential(rows):
    valid = [
        row for row in (rows or [])
        if row.get("score") is not None and row["score"] < 0
        and row.get("n_c_o_angle") is not None and 45 <= row["n_c_o_angle"] <= 155
    ]
    return closest(valid, "amine_to_acetyl_c_distance") or closest(rows, "amine_to_acetyl_c_distance")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rules = json.loads(RULES.read_text(encoding="utf-8"))
    structural = json.loads(STRUCTURES.read_text(encoding="utf-8"))
    docking = json.loads(DOCKING.read_text(encoding="utf-8"))
    rescue = {int(row["rank"]): row for row in json.loads(RESCUE.read_text(encoding="utf-8"))}
    rule_by_rank = {int(row["rank"]): row for row in rules["rows"]}
    structure_by_rank = {int(row["rank"]): row for row in structural}
    dock_by_rank = {int(row["rank"]): row for row in docking["candidates"]}

    rows = []
    for rank in sorted(rule_by_rank):
        rule = rule_by_rank[rank]
        structure = structure_by_rank.get(rank, {})
        dock = dock_by_rank.get(rank, {})
        dock_status = dock.get("docking_status", "")
        structure_record = dock.get("corrected_structure") or structure
        ac = closest(dock.get("acetyl_CoA_geometry"), "site_to_acetyl_c_distance")
        pap = closest(dock.get("pAP_pingpong_geometry"), "amine_to_site_distance")
        seq = best_sequential(dock.get("pAP_sequential_geometry"))
        rescue_row = rescue.get(rank, {})
        rescue_best = rescue_row.get("best") or {}

        ac_distance = ac.get("site_to_acetyl_c_distance")
        ac_angle = ac.get("site_c_o_angle")
        ac_score = ac.get("score")
        ac_source = "主搜索"
        if rescue_best:
            rescue_good_angle = 55 <= rescue_best.get("cys_c_o_angle_deg", -999) <= 150
            main_good_angle = ac_angle is not None and 55 <= ac_angle <= 150
            if rescue_good_angle and (not main_good_angle or rescue_best["cys_to_acetyl_c_A"] < (ac_distance or 999)):
                ac_distance = rescue_best["cys_to_acetyl_c_A"]
                ac_angle = rescue_best["cys_c_o_angle_deg"]
                ac_score = rescue_best["score"]
                ac_source = "结合位点中心补采样"

        if rule["forced_exclude"]:
            final_status = "强制排除"
            confidence = "排除"
            decision = rule["forced_exclusion_reason"]
        elif structure_record.get("transmembrane_count", 0) >= 4 or dock_status == "structural_exclude_multipass_membrane":
            final_status = "明显结构排除"
            confidence = "排除"
            decision = f"结构含{structure_record.get('transmembrane_count', 0)}段跨膜螺旋，缺少适合AcCoA和pAP共同进入的可溶性反应口袋。"
        elif rank in SELECTED_RANKS:
            final_status = "最终10个"
            confidence, decision = EVIDENCE[rank]
        elif dock_status in {"completed", "reused_previous"}:
            final_status = "结构可行备选"
            confidence = "C"
            decision = "至少存在局部双配体构象，但催化中心可信度、AcCoA反应端位置或取向弱于最终10个。"
        else:
            final_status = "未通过"
            confidence = "排除"
            decision = "未获得可用的结构/对接证据。"

        rows.append({
            "rank": rank, "y_pred": rule["y_pred"], "accession": rule["accession"],
            "protein_name": rule["protein_name"], "organism": rule["organism"],
            "pair_positive_4aminophenol": rule["pair_positive_4aminophenol"],
            "metaa": rule["metaa"], "metas": rule["metas"], "metas_highest_kept": rule["metas_highest_kept"],
            "forced_exclusion_reason": rule["forced_exclusion_reason"],
            "transmembrane_count": structure_record.get("transmembrane_count"),
            "hxxxd_count": structure_record.get("hxxxd_count"),
            "cys_triad_count": structure_record.get("cys_triad_count"),
            "ser_triad_count": structure_record.get("ser_triad_count"),
            "annotated_acyl_cys_count": len(structure_record.get("annotated_acyl_cys", [])),
            "structural_evidence": structure_record.get("structural_evidence", ""),
            "site_type": dock.get("site_type", ""), "docking_status": dock_status,
            "accoa_distance_A": ac_distance, "accoa_angle_deg": ac_angle, "accoa_score": ac_score,
            "accoa_geometry_source": ac_source if ac_distance is not None else "",
            "pap_to_site_A": pap.get("amine_to_site_distance"), "pap_score": pap.get("score"),
            "sequential_distance_A": seq.get("amine_to_acetyl_c_distance"),
            "sequential_angle_deg": seq.get("n_c_o_angle"), "sequential_score": seq.get("score"),
            "confidence": confidence, "final_status": final_status, "decision": decision,
            "structure_file": structure_record.get("structure_file", ""), "docking_folder": dock.get("folder", ""),
            "functional_annotation_for_postcheck": rule.get("uniprot_comments", ""),
        })

    selected = [row for row in rows if row["final_status"] == "最终10个"]
    backups = [row for row in rows if row["final_status"] == "结构可行备选"]
    summary = {
        "rank_block": "101-200",
        "dedup_retained": len(rows),
        "known_positive_excluded": sum(row["pair_positive_4aminophenol"] for row in rows),
        "metaa_count": sum(row["metaa"] for row in rows),
        "metas_count": sum(row["metas"] for row in rows),
        "metas_kept": {"rank": 112, "accession": "A8GKF8"},
        "forced_excluded_total": sum(row["final_status"] == "强制排除" for row in rows),
        "obvious_structural_excluded": sum(row["final_status"] == "明显结构排除" for row in rows),
        "selected_count": len(selected),
        "selected_ranks": [row["rank"] for row in selected],
        "selected_accessions": [row["accession"] for row in selected],
        "backup_count": len(backups),
        "functional_annotation_used_for_exclusion": False,
    }
    package = {"summary": summary, "selected": selected, "backups": backups, "rows": rows, "method": docking["method"]}
    (OUT / "corrected_final_assessment.json").write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(rows[0].keys())
    for name, data in (("corrected_selected_10.csv", selected), ("corrected_full_assessment.csv", rows), ("corrected_backups.csv", backups)):
        with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

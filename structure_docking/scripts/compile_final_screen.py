import csv
import json
from pathlib import Path


ROOT = Path(os.environ.get("DOCKING_ROOT", str(Path(__file__).resolve().parents[2])))
EMPIRICAL = ROOT / "07_mechanism_screen" / "empirical_screen.csv"
MOTIFS = ROOT / "07_mechanism_screen" / "motif_screen" / "motif_screen.csv"
DOCKING = ROOT / "07_mechanism_screen" / "mechanism_docking" / "mechanism_docking_to_rank_200.json"
OUT = ROOT / "08_mechanism_report"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value):
    return bool(value) and str(value).lower() not in {"false", "0", "none"}


def best_by(rows, key):
    rows = rows or []
    return min(rows, key=lambda row: row.get(key, float("inf"))) if rows else {}


def active_site_text(row):
    return row.get("features", "")


def is_exact_nat(row):
    name = row.get("protein_name", "").lower()
    return "arylamine n-acetyltransferase" in name and "arylalkylamine" not in name


def is_small_molecule_cys_acyltransferase(row):
    name = row.get("protein_name", "").lower()
    family = any(token in name for token in (
        "homoserine o-acetyltransferase",
        "homoserine o-succinyltransferase",
        "probable acyltransferase",
    ))
    return family and "acyl-thioester intermediate" in active_site_text(row).lower()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    empirical = read_csv(EMPIRICAL)
    motif_rows = read_csv(MOTIFS)
    motif_by_rank = {int(row["rank"]): row for row in motif_rows}
    docking = json.loads(DOCKING.read_text(encoding="utf-8"))
    dock_by_rank = {int(row["rank"]): row for row in docking["candidates"]}

    assessments = []
    for row in empirical:
        rank = int(row["rank"])
        motif = motif_by_rank.get(rank, {})
        dock = dock_by_rank.get(rank, {})
        ping = as_bool(dock.get("pingpong_pass"))
        seq_strict = as_bool(dock.get("sequential_strict_pass"))
        exact_nat = is_exact_nat(row)
        small_cys = is_small_molecule_cys_acyltransferase(row)
        best_ac = dock.get("best_reactive_acetyl_CoA") or {}
        best_pap = best_by(dock.get("pAP_pingpong_geometry"), "amine_to_site_distance")
        best_seq = best_by(dock.get("pAP_sequential_geometry"), "amine_to_acetyl_c_distance")

        disposition = row["empirical_disposition"]
        qualified = False
        route = ""
        confidence = "排除"
        decision = row["empirical_reason"]

        if disposition == "carry_to_structure_screen":
            if exact_nat and int(motif.get("triad_count") or 0) > 0:
                qualified = True
                route = "乒乓"
                if ping:
                    confidence = "A：天然反应直接相关＋完整三联体＋对接命中"
                    decision = "芳胺N-乙酰转移酶；UniProt标注酰基-硫酯中间体，结构中有完整Cys-His-Asp/Glu口袋，AcCoA与对氨基苯酚均命中催化区。"
                else:
                    confidence = "A-：天然反应直接相关＋完整三联体；AcCoA采样未命中"
                    decision = "芳胺N-乙酰转移酶天然反应与目标最接近，且UniProt标注催化半胱氨酸、结构三联体完整；对氨基苯酚可进入催化区，但本轮AcCoA构象未达到几何阈值，按对接假阴性/待复核记录。"
            elif small_cys and ping:
                qualified = True
                route = "乒乓"
                confidence = "B：小分子天然底物＋酰基半胱氨酸＋对接命中"
                decision = "天然受体为小分子；UniProt标注酰基-硫酯中间体，结构中存在Cys-His-Asp/Glu样口袋，AcCoA和对氨基苯酚定点对接均达到宽松反应几何。"
            elif (not small_cys) and seq_strict and (best_seq.get("score") is not None and best_seq.get("score") < 0):
                qualified = True
                route = "顺序"
                confidence = "C：双底物共同口袋计算支持"
                decision = "未采用乒乓证据；AcCoA固定后，对氨基苯酚在同一口袋达到严格距离/角度且Vina能量为负，列为顺序机制备选。"
            else:
                confidence = "未通过结构/机制门槛"
                decision = "未同时取得可接受的天然底物、催化基序/三联体和反应几何证据；不计入最终候选。"

        assessments.append({
            "rank": rank,
            "y_pred": float(row["y_pred"]),
            "accession": row["accession"],
            "protein_name": row["protein_name"],
            "organism": row["organism"],
            "empirical_disposition": disposition,
            "empirical_reason": row["empirical_reason"],
            "route": route,
            "qualified": qualified,
            "confidence": confidence,
            "decision": decision,
            "uniprot_active_sites": row.get("features", ""),
            "hxxxd_count": int(motif.get("hxxxd_count") or 0),
            "hxxxd_motifs": motif.get("hxxxd_motifs", ""),
            "triad_count": int(motif.get("triad_count") or 0),
            "triads": motif.get("cys_his_acid_triads", ""),
            "pingpong_docking_pass": ping,
            "accoa_mode": best_ac.get("mode"),
            "accoa_score": best_ac.get("score"),
            "cys_to_acetyl_c_A": best_ac.get("site_to_acetyl_c_distance"),
            "cys_c_o_angle_deg": best_ac.get("site_c_o_angle"),
            "pap_to_catalytic_site_A": best_pap.get("amine_to_site_distance"),
            "pap_pingpong_score": best_pap.get("score"),
            "sequential_strict_pass": seq_strict,
            "pap_n_to_acetyl_c_A": best_seq.get("amine_to_acetyl_c_distance"),
            "sequential_angle_deg": best_seq.get("n_c_o_angle"),
            "sequential_score": best_seq.get("score"),
            "structure_file": motif.get("structure_file", ""),
            "docking_folder": dock.get("folder", ""),
        })

    eligible = [row for row in assessments if row["qualified"]]
    eligible.sort(key=lambda row: row["rank"])
    priority = {"A": 0, "A-": 1, "B": 2, "C": 3}
    selected = sorted(
        eligible,
        key=lambda row: (priority.get(row["confidence"].split("：", 1)[0], 9), row["rank"]),
    )[:10]
    selected.sort(key=lambda row: row["rank"])
    selected_ranks = {row["rank"] for row in selected}
    for row in assessments:
        row["final_status"] = (
            "最终10个" if row["rank"] in selected_ranks
            else "合格备选" if row["qualified"]
            else "排除/未通过"
        )

    long_excluded = [row for row in assessments if row["empirical_disposition"] == "exclude_long_protein_or_polypeptide"]
    summary = {
        "screened_rank_block": "101-200",
        "extension_triggered": False,
        "deduplicated_retained": len(empirical),
        "long_protein_or_polypeptide_excluded": len(long_excluded),
        "qualified_total": len(eligible),
        "selected_count": len(selected),
        "selected_ranks": [row["rank"] for row in selected],
        "selected_accessions": [row["accession"] for row in selected],
        "selection_rule": "完成101-200整段后，先按机制证据等级选择（直接相关芳胺NAT优先，随后为小分子Cys酰基转移酶对接阳性），同等级内按原始排名；不足10才整段扩展201-300。",
    }

    package = {
        "summary": summary,
        "selected": selected,
        "eligible": eligible,
        "long_protein_or_polypeptide_excluded": long_excluded,
        "assessments": assessments,
        "docking_method": docking.get("method", {}),
    }
    (OUT / "final_screening_assessment.json").write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = list(assessments[0].keys())
    with (OUT / "full_mechanism_assessment.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(assessments)
    with (OUT / "selected_10.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    lines = [
        "# 101–200位机制筛选结论",
        "",
        f"完成90%去冗、UniProt天然底物经验筛选、AlphaFold结构基序/口袋检查及机制定点对接后，共有{len(eligible)}个候选达到综合门槛。最终先按机制证据等级选择、同等级内按原始排名，共取10个；101–200已足够，因此不扩展201–300。",
        "",
        "## 最终10个",
        "",
        "|排名|UniProt|蛋白|路线|证据等级|AcCoA几何|判断|",
        "|---:|---|---|---|---|---|---|",
    ]
    for item in selected:
        geometry = (
            f"{item['cys_to_acetyl_c_A']:.3f} Å / {item['cys_c_o_angle_deg']:.1f}°"
            if item["cys_to_acetyl_c_A"] is not None else "本轮未命中"
        )
        lines.append(f"|{item['rank']}|{item['accession']}|{item['protein_name']}|{item['route']}|{item['confidence']}|{geometry}|{item['decision']}|")
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 芳胺N-乙酰转移酶若UniProt明确标注芳胺反应、酰基硫酯中间体且结构三联体完整，即使单轮AcCoA构象采样未命中，也保留为A-级候选并明确标记，不把它写成对接阳性。",
        "- HAT/HST类只有天然底物为小分子、存在标注的酰基半胱氨酸并且定点对接达到宽松反应几何时才通过。",
        "- HXXXD本身不是充分条件；无乒乓口袋时，必须有双底物同口袋的严格顺序反应几何。",
        "- Vina和AlphaFold结果用于实验优先级排序，不等同于已证明的催化活性。",
        "",
        "## 长蛋白/多肽底物排除",
        "",
        "|排名|UniProt|蛋白|排除原因|",
        "|---:|---|---|---|",
    ])
    for item in long_excluded:
        lines.append(f"|{item['rank']}|{item['accession']}|{item['protein_name']}|{item['empirical_reason']}|")
    (OUT / "screening_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

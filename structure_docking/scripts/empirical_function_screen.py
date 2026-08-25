import csv
import json
from pathlib import Path


ROOT = Path(r"E:\total\docking_comparison\07_mechanism_screen")
SUMMARY = ROOT / "uniprot" / "uniprot_summary.json"

LONG_PROTEIN_RANKS = {
    103: "非催化性ADA组蛋白乙酰转移酶复合物组分；涉及蛋白质乙酰化体系",
    107: "DltC/磷壁酸大分子体系的D-丙氨酰转移，不是游离小分子乙酰受体",
    108: "原始受体为L-lysyl-[protein]，供体为octanoyl-[ACP]",
    110: "原始受体为L-lysyl-[protein]，供体为octanoyl-[ACP]",
    127: "原始受体为L-lysyl-[protein]，供体为octanoyl-[ACP]",
    146: "蛋白半胱氨酸棕榈酰化酶；原始受体为L-cysteinyl-[protein]",
    154: "原始受体为蛋白质N端氨基酸",
    156: "原始受体为L-lysyl-[protein]，供体为octanoyl-[ACP]",
    157: "RTX毒素蛋白赖氨酸酰化；原始受体为L-lysyl-[protein]",
    179: "原始受体为L-lysyl-[protein]，供体为octanoyl-[ACP]",
    186: "蛋白半胱氨酸棕榈酰化酶；原始受体为L-cysteinyl-[protein]",
    189: "原始受体为alpha-tubulin蛋白赖氨酸",
    198: "蛋白半胱氨酸棕榈酰化酶；原始受体为L-cysteinyl-[protein]",
}

UNRELATED_MEMBRANE_LIPID_RANKS = {
    135: "使用acyl phosphate的甘油-3-磷酸酰基转移酶；非AcCoA乙酰转移",
    142: "使用acyl phosphate的甘油-3-磷酸酰基转移酶；非AcCoA乙酰转移",
    143: "外膜PagP脂质A长链酰基转移酶；脂质底物/膜口袋与芳胺乙酰化不相符",
    167: "极长链脂肪酸延长酶；多跨膜且反应为碳链延长",
    169: "膜相关甘油-3-磷酸酰基转移酶；非AcCoA乙酰转移",
    172: "外膜PagP脂质A长链酰基转移酶；脂质底物/膜口袋与芳胺乙酰化不相符",
    175: "外膜PagP脂质A长链酰基转移酶；脂质底物/膜口袋与芳胺乙酰化不相符",
    178: "磷脂酰胆碱-甾醇长链酰基转移；底物与目标反应差异过大",
    182: "膜相关甘油-3-磷酸酰基转移酶；非AcCoA乙酰转移",
    184: "外膜PagP脂质A长链酰基转移酶；脂质底物/膜口袋与芳胺乙酰化不相符",
    193: "膜相关甘油-3-磷酸酰基转移酶；非AcCoA乙酰转移",
}


rows = json.loads(SUMMARY.read_text(encoding="utf-8"))
output = []
for row in sorted(rows, key=lambda item: int(item["rank"])):
    rank = int(row["rank"])
    if rank in LONG_PROTEIN_RANKS:
        disposition = "exclude_long_protein_or_polypeptide"
        reason = LONG_PROTEIN_RANKS[rank]
    elif rank in UNRELATED_MEMBRANE_LIPID_RANKS:
        disposition = "exclude_unrelated_membrane_or_lipid_reaction"
        reason = UNRELATED_MEMBRANE_LIPID_RANKS[rank]
    elif (row.get("identity_pct") or 0) < 99.9 or (row.get("query_coverage_pct") or 0) < 95:
        disposition = "defer_structure_mapping_not_exact"
        reason = f"UniProt结构映射并非原序列100%一致（identity={row.get('identity_pct')}%, coverage={row.get('query_coverage_pct')}%）"
    else:
        disposition = "carry_to_structure_screen"
        reason = "已知反应使用小分子底物，或功能尚未明确；继续检查催化基序与口袋"
    output.append({**row, "empirical_disposition": disposition, "empirical_reason": reason})

ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "empirical_screen.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
fields = [
    "rank", "y_pred", "accession", "protein_name", "organism", "sequence_length",
    "identity_pct", "query_coverage_pct", "ec_numbers", "empirical_disposition",
    "empirical_reason", "comments", "features", "sequence",
]
with (ROOT / "empirical_screen.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(output)

counts = {}
for row in output:
    counts[row["empirical_disposition"]] = counts.get(row["empirical_disposition"], 0) + 1
print(json.dumps(counts, ensure_ascii=False, indent=2))

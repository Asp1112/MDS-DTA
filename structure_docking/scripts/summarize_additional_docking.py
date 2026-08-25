import json
from pathlib import Path


path = Path(r"structure_docking/11_additional_6\07_docking\corrected_docking_results.json")
data = json.loads(path.read_text(encoding="utf-8"))
for row in data["candidates"]:
    if row.get("docking_status") != "completed":
        continue
    ac = row.get("best_relaxed_reactive_acetyl_CoA") or min(row["acetyl_CoA_geometry"], key=lambda x: x["site_to_acetyl_c_distance"])
    pp = min(row["pAP_pingpong_geometry"], key=lambda x: x["amine_to_site_distance"])
    sq = min(row["pAP_sequential_geometry"], key=lambda x: x["amine_to_acetyl_c_distance"])
    print(json.dumps({
        "rank": row["rank"], "accession": row["accession"], "site": row["site_type"],
        "ac_d": ac["site_to_acetyl_c_distance"], "ac_angle": ac["site_c_o_angle"], "ac_score": ac["score"],
        "pap_d": pp["amine_to_site_distance"], "pap_score": pp["score"],
        "seq_d": sq["amine_to_acetyl_c_distance"], "seq_angle": sq["n_c_o_angle"], "seq_score": sq["score"],
        "seq_pass": row.get("relaxed_sequential_pass"),
    }, ensure_ascii=False))

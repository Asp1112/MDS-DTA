import concurrent.futures
import json
import math
import sys
from pathlib import Path


ROOT = Path(r"E:\total\docking_comparison")
STRUCTURES = ROOT / "09_corrected_screen" / "structure_analysis" / "corrected_structure_analysis.json"
DOCKING = ROOT / "09_corrected_screen" / "docking"
ACCOA = Path(r"E:\autodockvina\acetyl_CoA.pdbqt")
TARGETS = {108, 127, 179}

sys.path.insert(0, str(ROOT))
from run_mechanism_docking import angle, parse_models, run_vina, write_config  # noqa: E402


records = {int(row["rank"]): row for row in json.loads(STRUCTURES.read_text(encoding="utf-8"))}
jobs = []
for rank in sorted(TARGETS):
    record = records[rank]
    sites = record.get("annotated_sites", [])
    if not sites:
        continue
    center = tuple(sum(site["center"][i] for site in sites) / len(sites) for i in range(3))
    folder = DOCKING / f"rank_{rank:04d}_{record['accession']}"
    receptor = folder / "receptor.pdbqt"
    config = folder / "acetyl_CoA.binding_centroid_rescue.config.txt"
    out = folder / "acetyl_CoA.binding_centroid_rescue.out.pdbqt"
    log = folder / "acetyl_CoA.binding_centroid_rescue.log.txt"
    write_config(config, receptor, ACCOA, center, 30.0)
    jobs.append({"rank": rank, "config": config, "out": out, "log": log, "error": folder / "acetyl_CoA.binding_centroid_rescue.error.txt"})

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    statuses = list(executor.map(run_vina, jobs))

results = []
for job in jobs:
    rank = job["rank"]
    record = records[rank]
    cys = tuple(record["annotated_acyl_cys"][0]["center"])
    geometry = []
    for model in parse_models(job["out"]):
        if not {"C2", "O1"}.issubset(model["atoms"]):
            continue
        d = math.dist(cys, model["atoms"]["C2"])
        a = angle(cys, model["atoms"]["C2"], model["atoms"]["O1"])
        geometry.append({"mode": model["mode"], "score": model["score"], "cys_to_acetyl_c_A": round(d, 3), "cys_c_o_angle_deg": round(a, 2)})
    geometry.sort(key=lambda row: (row["cys_to_acetyl_c_A"], row["score"] if row["score"] is not None else 99))
    results.append({"rank": rank, "accession": record["accession"], "geometry": geometry, "best": geometry[0] if geometry else None})

output = DOCKING / "acyl_cys_rescue_results.json"
output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps([{"rank": row["rank"], "accession": row["accession"], "best": row["best"]} for row in results], ensure_ascii=False, indent=2))

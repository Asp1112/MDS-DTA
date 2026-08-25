import concurrent.futures
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(os.environ.get("DOCKING_ROOT", str(Path(__file__).resolve().parents[2])))
ANALYSIS = ROOT / "09_corrected_screen" / "structure_analysis" / "corrected_structure_analysis.json"
OLD_RESULTS = ROOT / "07_mechanism_screen" / "mechanism_docking" / "mechanism_docking_to_rank_200.json"
OUT = ROOT / "09_corrected_screen" / "docking"
MGL_PYTHON = Path(r"E:\MGLTools\python.exe")
PREP_RECEPTOR = Path(r"E:\MGLTools\Lib\site-packages\AutoDockTools\Utilities24\prepare_receptor4.py")
PAP = Path(r"ligands/4_Aminophenol.pdbqt")
ACCOA = Path(r"E:\autodockvina\acetyl_CoA.pdbqt")

sys.path.insert(0, str(ROOT))
from run_mechanism_docking import (  # noqa: E402
    angle, complex_receptor, parse_models, parse_pdb_residues, run_vina, write_config,
)


REUSE_RANKS = {112, 141, 152, 195}


def geometric_center(residues):
    coords = [atoms["CA"] for atoms in residues.values() if "CA" in atoms]
    if not coords:
        return None
    return tuple(sum(row[i] for row in coords) / len(coords) for i in range(3))


def choose_site(record, residues):
    if record.get("annotated_acyl_cys"):
        site = record["annotated_acyl_cys"][0]
        return tuple(site["center"]), "annotated_acyl_cys", site
    if record.get("cys_triads"):
        site = record["cys_triads"][0]
        return tuple(site["center"]), "cys_triad", site
    if record.get("hxxxd_motifs"):
        pos = int(record["hxxxd_motifs"][0]["start"])
        for (chain, number, name), atoms in residues.items():
            if number == pos and name == "HIS":
                return atoms.get("NE2") or atoms.get("ND1") or atoms.get("CA"), "hxxxd", record["hxxxd_motifs"][0]
    if record.get("ser_triads"):
        site = record["ser_triads"][0]
        return tuple(site["center"]), "ser_triad", site
    if record.get("annotated_sites"):
        sites = record["annotated_sites"]
        center = tuple(sum(site["center"][i] for site in sites) / len(sites) for i in range(3))
        return center, "annotated_site_centroid", {"sites": sites}
    return geometric_center(residues), "geometric_center", {"reason": "无明确基序时用于宽松局部搜索"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    structural = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    old_payload = json.loads(OLD_RESULTS.read_text(encoding="utf-8"))
    old_by_rank = {int(row["rank"]): row for row in old_payload["candidates"]}
    final = []
    candidates = []
    accoa_jobs = []

    for record in structural:
        rank = int(record["rank"])
        if record.get("status") != "analyzed":
            final.append({**record, "docking_status": "structure_unavailable"})
            continue
        if record.get("transmembrane_count", 0) >= 4:
            final.append({**record, "docking_status": "structural_exclude_multipass_membrane"})
            continue
        if rank in REUSE_RANKS and rank in old_by_rank:
            final.append({**old_by_rank[rank], "docking_status": "reused_previous", "corrected_structure": record})
            continue

        folder = OUT / f"rank_{rank:04d}_{record['accession']}"
        folder.mkdir(parents=True, exist_ok=True)
        receptor_pdb = Path(record["structure_file"])
        receptor = folder / "receptor.pdbqt"
        if not receptor.exists():
            proc = subprocess.run(
                [str(MGL_PYTHON), str(PREP_RECEPTOR), "-r", str(receptor_pdb), "-o", str(receptor), "-A", "checkhydrogens"],
                capture_output=True, text=True, errors="replace",
            )
            (folder / "receptor_preparation.log.txt").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        if not receptor.exists():
            final.append({**record, "docking_status": "receptor_preparation_failed"})
            continue
        residues = parse_pdb_residues(receptor_pdb)
        center, site_type, site_detail = choose_site(record, residues)
        if center is None:
            final.append({**record, "docking_status": "no_search_center"})
            continue
        candidate = {
            "rank": rank, "accession": record["accession"], "protein_name": record["protein_name"],
            "site_type": site_type, "site_center": center, "site_detail": site_detail,
            "receptor": str(receptor), "folder": str(folder), "corrected_structure": record,
        }
        candidates.append(candidate)
        config = folder / "acetyl_CoA.targeted.config.txt"
        out = folder / "acetyl_CoA.targeted.out.pdbqt"
        log = folder / "acetyl_CoA.targeted.log.txt"
        write_config(config, receptor, ACCOA, center, 30.0)
        accoa_jobs.append({
            "rank": rank, "accession": record["accession"], "config": config, "out": out, "log": log,
            "error": folder / "acetyl_CoA.targeted.error.txt",
        })

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        accoa_status = list(executor.map(run_vina, accoa_jobs))
    accoa_by_rank = {job["rank"]: job for job in accoa_status}

    pap_jobs = []
    pending = []
    for candidate in candidates:
        rank = candidate["rank"]
        if accoa_by_rank.get(rank, {}).get("status") == "failed":
            final.append({**candidate, "docking_status": "accoa_failed"})
            continue
        folder = Path(candidate["folder"])
        accoa_models = parse_models(folder / "acetyl_CoA.targeted.out.pdbqt")
        if not accoa_models:
            final.append({**candidate, "docking_status": "accoa_no_models"})
            continue
        site = tuple(candidate["site_center"])
        geometry = []
        for model in accoa_models:
            if not {"C2", "O1"}.issubset(model["atoms"]):
                continue
            d = math.dist(site, model["atoms"]["C2"])
            a = angle(site, model["atoms"]["C2"], model["atoms"]["O1"])
            geometry.append({
                "mode": model["mode"], "score": model["score"],
                "site_to_acetyl_c_distance": round(d, 3), "site_c_o_angle": round(a, 2) if a is not None else None,
            })
        geometry.sort(key=lambda row: (row["site_to_acetyl_c_distance"], row["score"] if row["score"] is not None else 99))
        best_reactive = next((row for row in geometry if row["site_to_acetyl_c_distance"] <= 6.0 and 55 <= row["site_c_o_angle"] <= 150), None)
        selected_number = best_reactive["mode"] if best_reactive else min(accoa_models, key=lambda m: m["score"] if m["score"] is not None else 99)["mode"]
        selected = next(model for model in accoa_models if model["mode"] == selected_number)

        ping_config = folder / "4_aminophenol.pingpong.config.txt"
        ping_out = folder / "4_aminophenol.pingpong.out.pdbqt"
        write_config(ping_config, Path(candidate["receptor"]), PAP, site, 20.0)
        pap_jobs.append({"rank": rank, "kind": "pingpong", "config": ping_config, "out": ping_out, "log": folder / "4_aminophenol.pingpong.log.txt", "error": folder / "4_aminophenol.pingpong.error.txt"})

        complex_path = folder / "receptor_with_acetyl_CoA.pdbqt"
        complex_receptor(Path(candidate["receptor"]), selected, complex_path)
        acetyl_c = selected["atoms"]["C2"]
        seq_config = folder / "4_aminophenol.sequential.config.txt"
        seq_out = folder / "4_aminophenol.sequential.out.pdbqt"
        write_config(seq_config, complex_path, PAP, acetyl_c, 18.0)
        pap_jobs.append({"rank": rank, "kind": "sequential", "config": seq_config, "out": seq_out, "log": folder / "4_aminophenol.sequential.log.txt", "error": folder / "4_aminophenol.sequential.error.txt"})
        pending.append({
            **candidate, "docking_status": "accoa_finished", "acetyl_CoA_geometry": geometry,
            "best_relaxed_reactive_acetyl_CoA": best_reactive, "selected_acetyl_CoA_mode": selected_number,
            "selected_acetyl_CoA_score": selected["score"], "selected_acetyl_C": acetyl_c,
            "selected_carbonyl_O": selected["atoms"]["O1"],
        })

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        pap_status = list(executor.map(run_vina, pap_jobs))
    pap_by_key = {(job["rank"], job["kind"]): job for job in pap_status}

    for candidate in pending:
        rank = candidate["rank"]
        folder = Path(candidate["folder"])
        site = tuple(candidate["site_center"])
        ping_models = parse_models(folder / "4_aminophenol.pingpong.out.pdbqt") if pap_by_key.get((rank, "pingpong"), {}).get("status") != "failed" else []
        ping_geometry = []
        for model in ping_models:
            if "N1" in model["atoms"]:
                ping_geometry.append({"mode": model["mode"], "score": model["score"], "amine_to_site_distance": round(math.dist(model["atoms"]["N1"], site), 3)})
        ping_geometry.sort(key=lambda row: (row["amine_to_site_distance"], row["score"] if row["score"] is not None else 99))

        seq_models = parse_models(folder / "4_aminophenol.sequential.out.pdbqt") if pap_by_key.get((rank, "sequential"), {}).get("status") != "failed" else []
        seq_geometry = []
        for model in seq_models:
            if "N1" not in model["atoms"]:
                continue
            d = math.dist(model["atoms"]["N1"], tuple(candidate["selected_acetyl_C"]))
            a = angle(model["atoms"]["N1"], tuple(candidate["selected_acetyl_C"]), tuple(candidate["selected_carbonyl_O"]))
            seq_geometry.append({
                "mode": model["mode"], "score": model["score"], "amine_to_acetyl_c_distance": round(d, 3),
                "n_c_o_angle": round(a, 2) if a is not None else None,
            })
        seq_geometry.sort(key=lambda row: (row["amine_to_acetyl_c_distance"], row["score"] if row["score"] is not None else 99))
        cysteine_route = candidate["site_type"] in {"annotated_acyl_cys", "cys_triad"}
        candidate.update({
            "pAP_pingpong_geometry": ping_geometry,
            "pAP_sequential_geometry": seq_geometry,
            "relaxed_pingpong_pass": bool(cysteine_route and candidate.get("best_relaxed_reactive_acetyl_CoA") and any(row["amine_to_site_distance"] <= 10.0 for row in ping_geometry)),
            "relaxed_sequential_pass": any(row["amine_to_acetyl_c_distance"] <= 6.0 and 55 <= row["n_c_o_angle"] <= 150 and row["score"] is not None and row["score"] < 0 for row in seq_geometry),
            "docking_status": "completed",
        })
        final.append(candidate)

    final.sort(key=lambda row: int(row["rank"]))
    payload = {
        "method": {
            "screen": "corrected_relaxed_structure_only", "reused_previous_ranks": sorted(REUSE_RANKS),
            "multipass_membrane_exclusion": ">=4 annotated transmembrane helices", "accoa_box_A": 30,
            "pAP_pingpong_box_A": 20, "pAP_sequential_box_A": 18, "exhaustiveness": 8,
            "num_modes": 20, "seed": 20260806, "relaxed_distance_A": 6.0,
            "relaxed_angle_deg": [55, 150],
        },
        "candidates": final,
    }
    output = OUT / "corrected_docking_results.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "total": len(final), "reused": sum(row.get("docking_status") == "reused_previous" for row in final),
        "new_completed": sum(row.get("docking_status") == "completed" for row in final),
        "multipass_excluded": sum(row.get("docking_status") == "structural_exclude_multipass_membrane" for row in final),
        "relaxed_pingpong_pass_new": sum(bool(row.get("relaxed_pingpong_pass")) for row in final),
        "relaxed_sequential_pass_new": sum(bool(row.get("relaxed_sequential_pass")) for row in final),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

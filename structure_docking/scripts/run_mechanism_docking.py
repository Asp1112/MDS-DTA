import argparse
import concurrent.futures
import json
import math
import re
import subprocess
from pathlib import Path


ROOT = Path(r"structure_docking/07_mechanism_screen")
MOTIFS = ROOT / "motif_screen" / "motif_screen.json"
OUT = ROOT / "mechanism_docking"
MGL_PYTHON = Path(r"E:\MGLTools\python.exe")
PREP_RECEPTOR = Path(r"E:\MGLTools\Lib\site-packages\AutoDockTools\Utilities24\prepare_receptor4.py")
VINA = Path(r"vina")
PAP = Path(r"ligands/4_Aminophenol.pdbqt")
ACCOA = Path(r"ligands/acetyl_CoA.pdbqt")
SCORES = re.compile(r"REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)")


def coord(line):
    return (float(line[30:38]), float(line[38:46]), float(line[46:54]))


def parse_pdb_residues(path):
    residues = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            key = (line[21].strip() or "A", int(line[22:26]), line[17:20].strip())
            residues.setdefault(key, {})[line[12:16].strip()] = coord(line)
        except ValueError:
            continue
    return residues


def parse_models(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    scores = [float(value) for value in SCORES.findall(text)]
    models, current = [], []
    for line in text.splitlines():
        if line.startswith("MODEL"):
            current = []
        elif line.startswith("HETATM"):
            current.append(line)
        elif line.startswith("ENDMDL"):
            models.append(current)
            current = []
    if current:
        models.append(current)
    parsed = []
    for index, lines in enumerate(models):
        atoms = {line[12:16].strip(): coord(line) for line in lines}
        parsed.append({"mode": index + 1, "score": scores[index] if index < len(scores) else None, "atoms": atoms, "lines": lines})
    return parsed


def angle(a, vertex, c):
    va = [a[i] - vertex[i] for i in range(3)]
    vc = [c[i] - vertex[i] for i in range(3)]
    denom = math.sqrt(sum(x*x for x in va)) * math.sqrt(sum(x*x for x in vc))
    if not denom:
        return None
    cosine = max(-1.0, min(1.0, sum(va[i] * vc[i] for i in range(3)) / denom))
    return math.degrees(math.acos(cosine))


def write_config(path, receptor, ligand, center, size):
    path.write_text("\n".join([
        f"receptor = {receptor}", f"ligand = {ligand}", "",
        f"center_x = {center[0]:.3f}", f"center_y = {center[1]:.3f}", f"center_z = {center[2]:.3f}", "",
        f"size_x = {size:.3f}", f"size_y = {size:.3f}", f"size_z = {size:.3f}", "",
        "exhaustiveness = 8", "seed = 20260806", "num_modes = 20", "energy_range = 6", "",
    ]), encoding="ascii")


def run_vina(job):
    if job["out"].exists() and job["out"].stat().st_size > 0:
        return {**job, "status": "reused"}
    process = subprocess.run([
        str(VINA), "--config", str(job["config"]), "--out", str(job["out"]),
        "--log", str(job["log"]), "--cpu", "4",
    ], capture_output=True, text=True, errors="replace")
    if process.returncode != 0:
        job["error"].write_text(process.stdout + "\n" + process.stderr, encoding="utf-8")
    return {**job, "status": "finished" if process.returncode == 0 and job["out"].exists() else "failed"}


def choose_site(record, residues):
    triads = record["cys_his_acid_triads"]
    if triads:
        triad = triads[0]
        return tuple(triad["cys_coord"]), "cysteine_site", triad
    motifs = record["hxxxd_motifs"]
    if motifs:
        position = int(motifs[0]["sequence_start"])
        residue = next((atoms for (chain, number, name), atoms in residues.items() if number == position and name == "HIS"), None)
        if residue:
            center = residue.get("NE2") or residue.get("ND1") or residue.get("CA")
            return center, "hxxxd_site", motifs[0]
    return None, "no_site", None


def complex_receptor(receptor, ligand_mode, destination):
    receptor_lines = [line for line in receptor.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith(("ATOM", "HETATM", "TER"))]
    destination.write_text("\n".join(ligand_mode["lines"] + receptor_lines) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-max", type=int, default=141)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    records = [row for row in json.loads(MOTIFS.read_text(encoding="utf-8")) if int(row["rank"]) <= args.rank_max]
    candidates, accoa_jobs = [], []
    for record in records:
        rank, accession = int(record["rank"]), record["accession"]
        folder = OUT / f"rank_{rank:04d}_{accession}"
        folder.mkdir(parents=True, exist_ok=True)
        receptor_pdb = Path(record["structure_file"])
        receptor = folder / "receptor.pdbqt"
        if not receptor.exists():
            process = subprocess.run([str(MGL_PYTHON), str(PREP_RECEPTOR), "-r", str(receptor_pdb), "-o", str(receptor), "-A", "checkhydrogens"], capture_output=True, text=True, errors="replace")
            (folder / "receptor_preparation.log.txt").write_text(process.stdout + "\n" + process.stderr, encoding="utf-8")
        if not receptor.exists():
            candidates.append({"rank": rank, "accession": accession, "status": "receptor_preparation_failed"})
            continue
        residues = parse_pdb_residues(receptor_pdb)
        center, site_type, site_detail = choose_site(record, residues)
        candidate = {"rank": rank, "accession": accession, "protein_name": record["protein_name"], "site_type": site_type, "site_center": center, "site_detail": site_detail, "receptor": str(receptor), "folder": str(folder)}
        candidates.append(candidate)
        if center is None:
            continue
        config = folder / "acetyl_CoA.targeted.config.txt"
        out = folder / "acetyl_CoA.targeted.out.pdbqt"
        log = folder / "acetyl_CoA.targeted.log.txt"
        write_config(config, receptor, ACCOA, center, 30.0)
        accoa_jobs.append({"rank": rank, "accession": accession, "config": config, "out": out, "log": log, "error": folder / "acetyl_CoA.targeted.error.txt"})

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        accoa_status = list(executor.map(run_vina, accoa_jobs))
    status_by_rank = {job["rank"]: job for job in accoa_status}

    pap_jobs, analysis = [], []
    for candidate in candidates:
        rank = candidate["rank"]
        if candidate.get("site_center") is None or status_by_rank.get(rank, {}).get("status") == "failed":
            analysis.append({**candidate, "status": candidate.get("status", "no_dockable_site")})
            continue
        folder = Path(candidate["folder"])
        accoa_models = parse_models(folder / "acetyl_CoA.targeted.out.pdbqt")
        site = tuple(candidate["site_center"])
        geometry = []
        for model in accoa_models:
            if not {"C2", "O1"}.issubset(model["atoms"]):
                continue
            dist = math.dist(site, model["atoms"]["C2"])
            attack_angle = angle(site, model["atoms"]["C2"], model["atoms"]["O1"])
            geometry.append({"mode": model["mode"], "score": model["score"], "site_to_acetyl_c_distance": round(dist, 3), "site_c_o_angle": round(attack_angle, 2) if attack_angle is not None else None})
        geometry.sort(key=lambda row: (row["site_to_acetyl_c_distance"], row["score"] if row["score"] is not None else 99))
        best_reactive = next((row for row in geometry if row["site_to_acetyl_c_distance"] <= 5.0 and 70 <= row["site_c_o_angle"] <= 140), None)
        selected_mode_number = best_reactive["mode"] if best_reactive else min(accoa_models, key=lambda model: model["score"] if model["score"] is not None else 99)["mode"]
        selected_mode = next(model for model in accoa_models if model["mode"] == selected_mode_number)

        ping_config = folder / "4_aminophenol.pingpong.config.txt"
        ping_out = folder / "4_aminophenol.pingpong.out.pdbqt"
        write_config(ping_config, Path(candidate["receptor"]), PAP, site, 18.0)
        pap_jobs.append({"rank": rank, "kind": "pingpong", "config": ping_config, "out": ping_out, "log": folder / "4_aminophenol.pingpong.log.txt", "error": folder / "4_aminophenol.pingpong.error.txt"})

        complex_path = folder / "receptor_with_acetyl_CoA.pdbqt"
        complex_receptor(Path(candidate["receptor"]), selected_mode, complex_path)
        acetyl_c = selected_mode["atoms"]["C2"]
        seq_config = folder / "4_aminophenol.sequential.config.txt"
        seq_out = folder / "4_aminophenol.sequential.out.pdbqt"
        write_config(seq_config, complex_path, PAP, acetyl_c, 15.0)
        pap_jobs.append({"rank": rank, "kind": "sequential", "config": seq_config, "out": seq_out, "log": folder / "4_aminophenol.sequential.log.txt", "error": folder / "4_aminophenol.sequential.error.txt"})
        analysis.append({**candidate, "status": "accoa_finished", "acetyl_CoA_geometry": geometry, "best_reactive_acetyl_CoA": best_reactive, "selected_acetyl_CoA_mode": selected_mode_number, "selected_acetyl_CoA_score": selected_mode["score"], "selected_acetyl_C": acetyl_c, "selected_carbonyl_O": selected_mode["atoms"]["O1"]})

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        pap_status = list(executor.map(run_vina, pap_jobs))
    pap_status_by_key = {(job["rank"], job["kind"]): job for job in pap_status}

    for candidate in analysis:
        rank = candidate["rank"]
        if candidate.get("status") != "accoa_finished":
            continue
        site = tuple(candidate["site_center"])
        ping_models = parse_models(Path(candidate["folder"]) / "4_aminophenol.pingpong.out.pdbqt") if pap_status_by_key[(rank, "pingpong")]["status"] != "failed" else []
        ping_geometry = []
        for model in ping_models:
            if "N1" in model["atoms"]:
                ping_geometry.append({"mode": model["mode"], "score": model["score"], "amine_to_site_distance": round(math.dist(model["atoms"]["N1"], site), 3)})
        ping_geometry.sort(key=lambda row: (row["amine_to_site_distance"], row["score"] if row["score"] is not None else 99))
        candidate["pAP_pingpong_geometry"] = ping_geometry
        candidate["pingpong_pass"] = bool(candidate.get("best_reactive_acetyl_CoA") and any(row["amine_to_site_distance"] <= 8.0 for row in ping_geometry))

        seq_models = parse_models(Path(candidate["folder"]) / "4_aminophenol.sequential.out.pdbqt") if pap_status_by_key[(rank, "sequential")]["status"] != "failed" else []
        seq_geometry = []
        for model in seq_models:
            if "N1" not in model["atoms"]:
                continue
            dist = math.dist(model["atoms"]["N1"], tuple(candidate["selected_acetyl_C"]))
            attack_angle = angle(model["atoms"]["N1"], tuple(candidate["selected_acetyl_C"]), tuple(candidate["selected_carbonyl_O"]))
            seq_geometry.append({"mode": model["mode"], "score": model["score"], "amine_to_acetyl_c_distance": round(dist, 3), "n_c_o_angle": round(attack_angle, 2) if attack_angle is not None else None})
        seq_geometry.sort(key=lambda row: (row["amine_to_acetyl_c_distance"], row["score"] if row["score"] is not None else 99))
        candidate["pAP_sequential_geometry"] = seq_geometry
        candidate["sequential_strict_pass"] = any(row["amine_to_acetyl_c_distance"] < 4.0 and 80 <= row["n_c_o_angle"] <= 130 for row in seq_geometry)
        candidate["sequential_lenient_pass"] = any(row["amine_to_acetyl_c_distance"] <= 5.0 and 70 <= row["n_c_o_angle"] <= 140 for row in seq_geometry)
        candidate["status"] = "completed"

    payload = {"method": {"rank_max": args.rank_max, "accoa_box_A": 30, "pingpong_pAP_box_A": 18, "sequential_pAP_box_A": 15, "exhaustiveness": 8, "num_modes": 20, "seed": 20260806, "strict_attack_distance_A": 4.0, "lenient_attack_distance_A": 5.0}, "candidates": analysis}
    (OUT / f"mechanism_docking_to_rank_{args.rank_max}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidates": len(analysis), "completed": sum(row.get("status") == "completed" for row in analysis), "pingpong_pass": sum(bool(row.get("pingpong_pass")) for row in analysis), "sequential_strict_pass": sum(bool(row.get("sequential_strict_pass")) for row in analysis), "sequential_lenient_pass": sum(bool(row.get("sequential_lenient_pass")) for row in analysis)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

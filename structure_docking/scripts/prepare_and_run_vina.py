import argparse
import concurrent.futures
import csv
import json
import re
import subprocess
from pathlib import Path


VINA_RESULT = re.compile(r"^REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)", re.MULTILINE)


def receptor_bounds(path):
    coordinates = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                coordinates.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue
    if not coordinates:
        raise ValueError(f"No receptor coordinates in {path}")
    axes = list(zip(*coordinates))
    minima = [min(axis) for axis in axes]
    maxima = [max(axis) for axis in axes]
    center = [(lo + hi) / 2 for lo, hi in zip(minima, maxima)]
    extent = [hi - lo for lo, hi in zip(minima, maxima)]
    return center, extent


def bounded_box(extent, padding=8.0, minimum=30.0, maximum=50.0):
    return [round(min(max(value + padding, minimum), maximum), 3) for value in extent]


def write_config(path, receptor, ligand, center, size, exhaustiveness, seed, num_modes):
    text = "\n".join(
        [
            f"receptor = {receptor}",
            f"ligand = {ligand}",
            "",
            f"center_x = {center[0]:.3f}",
            f"center_y = {center[1]:.3f}",
            f"center_z = {center[2]:.3f}",
            "",
            f"size_x = {size[0]:.3f}",
            f"size_y = {size[1]:.3f}",
            f"size_z = {size[2]:.3f}",
            "",
            f"exhaustiveness = {exhaustiveness}",
            f"seed = {seed}",
            f"num_modes = {num_modes}",
            "energy_range = 5",
            "",
        ]
    )
    path.write_text(text, encoding="ascii")


def run_one(job):
    if job["output"].exists() and job["output"].stat().st_size > 0:
        status = "reused"
        log_text = job["log"].read_text(encoding="utf-8", errors="replace") if job["log"].exists() else ""
    else:
        command = [
            str(job["vina"]),
            "--config",
            str(job["config"]),
            "--out",
            str(job["output"]),
            "--log",
            str(job["log"]),
            "--cpu",
            str(job["cpu"]),
        ]
        process = subprocess.run(command, capture_output=True, text=True, errors="replace")
        status = "finished" if process.returncode == 0 and job["output"].exists() else "failed"
        log_text = "\n".join([process.stdout, process.stderr])
        if process.returncode != 0:
            job["error_file"].write_text(log_text, encoding="utf-8")
    output_text = job["output"].read_text(encoding="utf-8", errors="replace") if job["output"].exists() else ""
    scores = [float(value) for value in VINA_RESULT.findall(output_text)]
    return {
        "rank": job["rank"],
        "accession": job["accession"],
        "ligand": job["ligand_name"],
        "status": status,
        "best_affinity_kcal_mol": min(scores) if scores else None,
        "mode_count": len(scores),
        "center_x": job["center"][0],
        "center_y": job["center"][1],
        "center_z": job["center"][2],
        "size_x": job["size"][0],
        "size_y": job["size"][1],
        "size_z": job["size"][2],
        "receptor_pdbqt": str(job["receptor"]),
        "config_file": str(job["config"]),
        "output_pdbqt": str(job["output"]),
        "log_file": str(job["log"]),
        "diagnostic": log_text[-1000:] if status == "failed" else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure-manifest", default=r"E:\total\docking_comparison\04_alphafold_structures\structure_manifest.json")
    parser.add_argument("--output-dir", default=r"E:\total\docking_comparison\05_vina")
    parser.add_argument("--mgl-python", default=r"E:\MGLTools\python.exe")
    parser.add_argument("--prepare-receptor", default=r"E:\MGLTools\Lib\site-packages\AutoDockTools\Utilities24\prepare_receptor4.py")
    parser.add_argument("--vina", default=r"E:\autodock\vina.exe")
    parser.add_argument("--pap-ligand", default=r"E:\total\4_Aminophenol.pdbqt")
    parser.add_argument("--accoa-ligand", default=r"E:\total\acetyl_CoA.pdbqt")
    parser.add_argument("--parallel-jobs", type=int, default=2)
    parser.add_argument("--cpu-per-job", type=int, default=5)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--num-modes", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()

    manifest = json.loads(Path(args.structure_manifest).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    preparation = []
    for record in manifest["records"]:
        if record.get("status") != "downloaded":
            continue
        rank = int(record["rank"])
        accession = record["accession"]
        candidate_dir = output_dir / f"rank_{rank:04d}_{accession}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        receptor_pdb = Path(record["pdb_file"])
        receptor_pdbqt = candidate_dir / f"rank_{rank:04d}_{accession}_receptor.pdbqt"
        prep_log = candidate_dir / "receptor_preparation.log.txt"
        if not receptor_pdbqt.exists():
            command = [
                args.mgl_python,
                args.prepare_receptor,
                "-r",
                str(receptor_pdb),
                "-o",
                str(receptor_pdbqt),
                "-A",
                "checkhydrogens",
            ]
            process = subprocess.run(command, capture_output=True, text=True, errors="replace")
            prep_log.write_text("\n".join([process.stdout, process.stderr]), encoding="utf-8")
            if process.returncode != 0 or not receptor_pdbqt.exists():
                preparation.append({"rank": rank, "accession": accession, "status": "failed", "log": str(prep_log)})
                continue
        preparation.append({"rank": rank, "accession": accession, "status": "prepared", "receptor_pdbqt": str(receptor_pdbqt)})
        center, extent = receptor_bounds(receptor_pdbqt)
        for ligand_name, ligand_path in (("4_aminophenol", Path(args.pap_ligand)), ("acetyl_CoA", Path(args.accoa_ligand))):
            size = bounded_box(extent, maximum=48.0 if ligand_name == "4_aminophenol" else 36.0)
            config = candidate_dir / f"{ligand_name}.config.txt"
            output = candidate_dir / f"{ligand_name}.out.pdbqt"
            log = candidate_dir / f"{ligand_name}.vina.log.txt"
            write_config(config, receptor_pdbqt, ligand_path, center, size, args.exhaustiveness, args.seed, args.num_modes)
            jobs.append(
                {
                    "rank": rank,
                    "accession": accession,
                    "ligand_name": ligand_name,
                    "ligand": ligand_path,
                    "receptor": receptor_pdbqt,
                    "config": config,
                    "output": output,
                    "log": log,
                    "error_file": candidate_dir / f"{ligand_name}.error.txt",
                    "vina": Path(args.vina),
                    "cpu": args.cpu_per_job,
                    "center": center,
                    "size": size,
                }
            )

    (output_dir / "receptor_preparation.json").write_text(json.dumps(preparation, ensure_ascii=False, indent=2), encoding="utf-8")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_jobs) as executor:
        for result in executor.map(run_one, jobs):
            results.append(result)
            (output_dir / "docking_results.partial.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    results.sort(key=lambda row: (row["rank"], row["ligand"]))
    payload = {
        "method": {
            "vina_executable": args.vina,
            "receptor_preparation": "AutoDockTools prepare_receptor4.py -A checkhydrogens; default cleanup and Gasteiger charges",
            "grid": "receptor bounding-box center; each dimension = receptor extent + 8 A; bounded to 30-48 A for 4-aminophenol and 30-36 A for acetyl-CoA",
            "exhaustiveness": args.exhaustiveness,
            "num_modes": args.num_modes,
            "energy_range_kcal_mol": 5,
            "seed": args.seed,
            "parallel_jobs": args.parallel_jobs,
            "cpu_per_job": args.cpu_per_job,
        },
        "results": results,
    }
    (output_dir / "docking_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "docking_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = list(results[0].keys()) if results else ["rank", "accession", "ligand", "status", "best_affinity_kcal_mol"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({"prepared": sum(r["status"] == "prepared" for r in preparation), "docking_jobs": len(results), "failed": sum(r["status"] == "failed" for r in results)}, indent=2))


if __name__ == "__main__":
    main()

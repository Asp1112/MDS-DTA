import json
import math
import re
from pathlib import Path


ROOT = Path(r"E:\autodockvina")
REFERENCES = ["metAA", "O31995", "Q8P051", "Q9KL03", "P26825", "nhoa"]


def atoms_by_model(path):
    models = []
    current = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MODEL"):
            current = []
        elif line.startswith(("ATOM", "HETATM")):
            try:
                current.append({
                    "name": line[12:16].strip(), "resname": line[17:20].strip(),
                    "resnum": int(line[22:26]),
                    "coord": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                })
            except ValueError:
                pass
        elif line.startswith("ENDMDL"):
            models.append(current)
            current = []
    if current:
        models.append(current)
    return models


def affinity(path):
    return [float(value) for value in re.findall(r"REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)", path.read_text(encoding="utf-8", errors="replace"))]


results = []
for name in REFERENCES:
    receptor = atoms_by_model(ROOT / f"{name}.pdbqt")[0]
    cysteines = [atom for atom in receptor if atom["resname"] == "CYS" and atom["name"] == "SG"]
    for ligand, atom_name, suffix in (("acetyl_CoA", "C2", "acetyl_CoA_out.pdbqt"), ("4_aminophenol", "N1", "out.pdbqt")):
        path = ROOT / f"{name}_{suffix}" if name != "metAA" or ligand == "acetyl_CoA" else ROOT / "metAA_out.pdbqt"
        if not path.exists():
            continue
        scores = affinity(path)
        for index, model in enumerate(atoms_by_model(path)):
            target = next((atom for atom in model if atom["name"] == atom_name), None)
            if not target:
                continue
            distances = sorted((math.dist(target["coord"], cys["coord"]), cys["resnum"]) for cys in cysteines)
            results.append({
                "reference": name, "ligand": ligand, "mode": index + 1,
                "affinity": scores[index] if index < len(scores) else None,
                "nearest_cys_distance": round(distances[0][0], 3) if distances else None,
                "nearest_cys_resnum": distances[0][1] if distances else None,
            })

Path(r"E:\total\docking_comparison\07_mechanism_screen\reference_docking_geometry.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
for name in REFERENCES:
    subset = [row for row in results if row["reference"] == name and row["ligand"] == "acetyl_CoA"]
    if subset:
        print(name, sorted(subset, key=lambda row: row["nearest_cys_distance"])[:3])

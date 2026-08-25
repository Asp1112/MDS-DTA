import csv
import json
import math
import re
from pathlib import Path

import numpy as np


ROOT = Path(r"E:\total\docking_comparison\07_mechanism_screen")
MANIFEST = ROOT / "structures" / "structure_manifest.json"
EMPIRICAL = ROOT / "empirical_screen.json"
OUT = ROOT / "motif_screen"
OUT.mkdir(parents=True, exist_ok=True)

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80}


def distance(a, b):
    return math.dist(a, b)


def fibonacci_sphere(samples=240):
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(samples):
        y = 1 - (index / float(samples - 1)) * 2
        radius = math.sqrt(max(0.0, 1 - y * y))
        theta = phi * index
        points.append((math.cos(theta) * radius, y, math.sin(theta) * radius))
    return points


SPHERE = fibonacci_sphere()


def sulfur_accessibility(sg, atoms):
    probe = 1.4
    sulfur_radius = VDW["S"] + probe
    points = np.asarray(SPHERE, dtype=float) * sulfur_radius + np.asarray(sg, dtype=float)
    coordinates = np.asarray([atom["coord"] for atom in atoms], dtype=float)
    radii = np.asarray([VDW.get(atom["element"], 1.75) + probe for atom in atoms], dtype=float)
    distances2 = np.sum((points[:, None, :] - coordinates[None, :, :]) ** 2, axis=2)
    blocked = np.any(distances2 < (radii[None, :] ** 2 - 1e-6), axis=1)
    return round(float(np.mean(~blocked)), 4)


def parse_pdb(path):
    atoms = []
    residues = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        chain = line[21].strip() or "A"
        try:
            residue_number = int(line[22:26])
            coord = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            bfactor = float(line[60:66])
        except ValueError:
            continue
        element = (line[76:78].strip() or re.sub(r"[^A-Za-z]", "", atom_name)[:1]).upper()
        atom = {
            "name": atom_name, "resname": residue_name, "chain": chain,
            "resnum": residue_number, "coord": coord, "bfactor": bfactor,
            "element": element,
        }
        atoms.append(atom)
        residues.setdefault((chain, residue_number, residue_name), {})[atom_name] = atom
    ordered = sorted(residues, key=lambda key: (key[0], key[1]))
    sequence = "".join(AA3.get(key[2], "X") for key in ordered)
    index_to_key = {index + 1: key for index, key in enumerate(ordered)}
    return atoms, residues, sequence, index_to_key


empirical = {int(row["rank"]): row for row in json.loads(EMPIRICAL.read_text(encoding="utf-8"))}
results = []
for structure in json.loads(MANIFEST.read_text(encoding="utf-8")):
    if structure["status"] != "downloaded":
        continue
    rank = int(structure["rank"])
    atoms, residues, sequence, index_to_key = parse_pdb(Path(structure["pdb_file"]))
    hxxxd = []
    for match in re.finditer(r"H...D", sequence):
        hxxxd.append({"sequence_start": match.start() + 1, "motif": match.group()})
    cys_sites = []
    histidines = []
    acids = []
    for key, residue_atoms in residues.items():
        chain, resnum, resname = key
        if resname == "CYS" and "SG" in residue_atoms:
            sg = residue_atoms["SG"]
            cys_sites.append({
                "chain": chain, "resnum": resnum, "coord": sg["coord"],
                "plddt": sg["bfactor"], "relative_sg_accessibility": sulfur_accessibility(sg["coord"], atoms),
            })
        elif resname == "HIS":
            for atom_name in ("ND1", "NE2"):
                if atom_name in residue_atoms:
                    histidines.append((chain, resnum, atom_name, residue_atoms[atom_name]))
        elif resname in {"ASP", "GLU"}:
            names = ("OD1", "OD2") if resname == "ASP" else ("OE1", "OE2")
            for atom_name in names:
                if atom_name in residue_atoms:
                    acids.append((chain, resnum, resname, atom_name, residue_atoms[atom_name]))

    triads = []
    for cys in cys_sites:
        for h_chain, h_resnum, h_atom_name, h_atom in histidines:
            ch = distance(cys["coord"], h_atom["coord"])
            if ch > 5.0:
                continue
            best_acid = None
            for a_chain, a_resnum, a_resname, a_atom_name, a_atom in acids:
                if a_resnum in {cys["resnum"], h_resnum}:
                    continue
                ha = distance(h_atom["coord"], a_atom["coord"])
                if ha <= 5.0 and (best_acid is None or ha < best_acid["his_acid_distance"]):
                    best_acid = {
                        "acid": f"{a_resname}{a_resnum}:{a_atom_name}",
                        "his_acid_distance": round(ha, 3),
                    }
            if best_acid:
                triads.append({
                    "cys": f"CYS{cys['resnum']}:SG", "his": f"HIS{h_resnum}:{h_atom_name}",
                    "cys_his_distance": round(ch, 3), **best_acid,
                    "cys_accessibility": cys["relative_sg_accessibility"],
                    "cys_coord": cys["coord"],
                })
    triads.sort(key=lambda item: item["cys_his_distance"] + item["his_acid_distance"])
    exposed = [site for site in cys_sites if site["relative_sg_accessibility"] >= 0.05 and site["plddt"] >= 70]
    row = empirical[rank]
    results.append({
        "rank": rank, "accession": structure["accession"], "protein_name": row["protein_name"],
        "sequence_length": len(sequence), "hxxxd_motifs": hxxxd, "hxxxd_count": len(hxxxd),
        "cysteines": cys_sites, "cysteine_count": len(cys_sites),
        "exposed_cysteines": exposed, "exposed_cysteine_count": len(exposed),
        "cys_his_acid_triads": triads, "triad_count": len(triads),
        "structure_file": structure["pdb_file"],
    })

(OUT / "motif_screen.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
with (OUT / "motif_screen.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    fields = ["rank", "accession", "protein_name", "sequence_length", "hxxxd_count", "hxxxd_motifs", "cysteine_count", "exposed_cysteine_count", "exposed_cysteines", "triad_count", "cys_his_acid_triads", "structure_file"]
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for result in results:
        writer.writerow({**result, "hxxxd_motifs": json.dumps(result["hxxxd_motifs"]), "exposed_cysteines": json.dumps(result["exposed_cysteines"]), "cys_his_acid_triads": json.dumps(result["cys_his_acid_triads"])})
print(json.dumps({"structures": len(results), "with_hxxxd": sum(r["hxxxd_count"] > 0 for r in results), "with_exposed_cys": sum(r["exposed_cysteine_count"] > 0 for r in results), "with_triad": sum(r["triad_count"] > 0 for r in results)}, ensure_ascii=False))

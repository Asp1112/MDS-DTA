import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(os.environ.get("DOCKING_ROOT", str(Path(__file__).resolve().parents[2])))
RULES = ROOT / "09_corrected_screen" / "corrected_rule_application.json"
MANIFEST = ROOT / "09_corrected_screen" / "structures" / "structure_manifest.json"
OUT = ROOT / "09_corrected_screen" / "structure_analysis"
OUT.mkdir(parents=True, exist_ok=True)

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def dist(a, b):
    return math.dist(a, b)


def parse_pdb(path):
    atoms = []
    residues = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            atom = {
                "name": line[12:16].strip(), "resname": line[17:20].strip(),
                "chain": line[21].strip() or "A", "resnum": int(line[22:26]),
                "coord": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                "plddt": float(line[60:66]),
            }
        except ValueError:
            continue
        atoms.append(atom)
        residues.setdefault((atom["chain"], atom["resnum"], atom["resname"]), {})[atom["name"]] = atom
    ordered = sorted(residues, key=lambda key: (key[0], key[1]))
    sequence = "".join(AA3.get(key[2], "X") for key in ordered)
    return atoms, residues, sequence


def atom_for(residues, resnum, preferred):
    for (chain, number, resname), atoms in residues.items():
        if number != resnum:
            continue
        for name in preferred:
            if name in atoms:
                return atoms[name]
        if "CA" in atoms:
            return atoms["CA"]
    return None


def triads(residues, nucleophile_name):
    nucleophiles, histidines, acids = [], [], []
    for (chain, number, resname), atoms in residues.items():
        if resname == nucleophile_name:
            atom_name = "SG" if resname == "CYS" else "OG"
            if atom_name in atoms:
                nucleophiles.append((number, atom_name, atoms[atom_name]))
        elif resname == "HIS":
            for atom_name in ("ND1", "NE2"):
                if atom_name in atoms:
                    histidines.append((number, atom_name, atoms[atom_name]))
        elif resname in {"ASP", "GLU"}:
            for atom_name in (("OD1", "OD2") if resname == "ASP" else ("OE1", "OE2")):
                if atom_name in atoms:
                    acids.append((number, resname, atom_name, atoms[atom_name]))
    found = []
    for n_num, n_atom_name, n_atom in nucleophiles:
        for h_num, h_atom_name, h_atom in histidines:
            nh = dist(n_atom["coord"], h_atom["coord"])
            if nh > 6.0:
                continue
            nearest = None
            for a_num, a_resname, a_atom_name, a_atom in acids:
                if a_num in {n_num, h_num}:
                    continue
                ha = dist(h_atom["coord"], a_atom["coord"])
                if ha <= 6.0 and (nearest is None or ha < nearest["his_acid_distance"]):
                    nearest = {"acid": f"{a_resname}{a_num}:{a_atom_name}", "his_acid_distance": round(ha, 3)}
            if nearest:
                found.append({
                    "nucleophile": f"{nucleophile_name}{n_num}:{n_atom_name}",
                    "his": f"HIS{h_num}:{h_atom_name}", "nucleophile_his_distance": round(nh, 3),
                    **nearest, "center": n_atom["coord"], "plddt": n_atom["plddt"],
                })
    found.sort(key=lambda row: row["nucleophile_his_distance"] + row["his_acid_distance"])
    return found


rules = json.loads(RULES.read_text(encoding="utf-8"))
rule_by_rank = {int(row["rank"]): row for row in rules["rows"]}
manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
results = []
for item in manifest:
    rank = int(item["rank"])
    source = rule_by_rank[rank]
    if item["status"] != "downloaded":
        results.append({"rank": rank, "accession": item["accession"], "status": "structure_failed", "reason": item.get("reason", "")})
        continue
    atoms, residues, sequence = parse_pdb(Path(item["pdb_file"]))
    hxxxd = [{"start": m.start() + 1, "motif": m.group()} for m in re.finditer(r"H...D", sequence)]
    cys_triads = triads(residues, "CYS")
    ser_triads = triads(residues, "SER")
    active_positions = [int(x) for x in re.findall(r"Active site:(\d+)-", source.get("uniprot_features", ""))]
    binding_positions = [int(x) for x in re.findall(r"Binding site:(\d+)-", source.get("uniprot_features", ""))]
    annotated_acyl_cys = []
    for match in re.finditer(r"Active site:(\d+)-\d+:Acyl-thioester intermediate", source.get("uniprot_features", ""), re.I):
        pos = int(match.group(1))
        atom = atom_for(residues, pos, ("SG",))
        if atom and atom["resname"] == "CYS":
            annotated_acyl_cys.append({"residue": f"CYS{pos}", "center": atom["coord"], "plddt": atom["plddt"]})
    annotated_atoms = []
    for pos in sorted(set(active_positions + binding_positions)):
        atom = atom_for(residues, pos, ("SG", "OG", "ND1", "NE2", "NZ", "CA"))
        if atom:
            annotated_atoms.append({"residue": f"{atom['resname']}{pos}", "center": atom["coord"], "plddt": atom["plddt"]})
    transmembrane_count = len(re.findall(r"Transmembrane:", source.get("uniprot_features", "")))
    mean_plddt = round(sum(a["plddt"] for a in atoms if a["name"] == "CA") / max(1, sum(a["name"] == "CA" for a in atoms)), 2)
    evidence = []
    if annotated_acyl_cys:
        evidence.append("UniProt标注酰基硫酯Cys")
    if cys_triads:
        evidence.append("空间Cys-His-Asp/Glu样口袋")
    if ser_triads:
        evidence.append("空间Ser-His-Asp/Glu样口袋")
    if hxxxd:
        evidence.append("HXXXD基序")
    if annotated_atoms:
        evidence.append("有注释催化/结合位点")
    if transmembrane_count >= 4:
        evidence.append(f"多跨膜({transmembrane_count})")
    structural_priority = (
        0 if annotated_acyl_cys else
        1 if cys_triads else
        2 if hxxxd or ser_triads else
        3 if annotated_atoms else 4
    )
    results.append({
        "rank": rank, "y_pred": source["y_pred"], "accession": item["accession"],
        "protein_name": source["protein_name"], "organism": source["organism"], "status": "analyzed",
        "sequence_length": len(sequence), "mean_plddt": mean_plddt, "transmembrane_count": transmembrane_count,
        "hxxxd_motifs": hxxxd, "hxxxd_count": len(hxxxd), "cys_triads": cys_triads,
        "cys_triad_count": len(cys_triads), "ser_triads": ser_triads, "ser_triad_count": len(ser_triads),
        "annotated_acyl_cys": annotated_acyl_cys, "annotated_sites": annotated_atoms,
        "structural_evidence": "；".join(evidence), "structural_priority": structural_priority,
        "structure_file": item["pdb_file"], "structure_source": item.get("source", ""),
    })

(OUT / "corrected_structure_analysis.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
fields = [
    "rank", "y_pred", "accession", "protein_name", "organism", "status", "sequence_length", "mean_plddt",
    "transmembrane_count", "hxxxd_count", "hxxxd_motifs", "cys_triad_count", "cys_triads", "ser_triad_count",
    "ser_triads", "annotated_acyl_cys", "annotated_sites", "structural_evidence", "structural_priority",
    "structure_file", "structure_source",
]
with (OUT / "corrected_structure_analysis.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in results:
        writer.writerow({**row, **{key: json.dumps(row.get(key, []), ensure_ascii=False) for key in (
            "hxxxd_motifs", "cys_triads", "ser_triads", "annotated_acyl_cys", "annotated_sites")}})
print(json.dumps({
    "records": len(results), "analyzed": sum(r["status"] == "analyzed" for r in results),
    "with_acyl_cys": sum(bool(r.get("annotated_acyl_cys")) for r in results),
    "with_cys_triad": sum(r.get("cys_triad_count", 0) > 0 for r in results),
    "with_hxxxd": sum(r.get("hxxxd_count", 0) > 0 for r in results),
    "multi_pass_membrane": sum(r.get("transmembrane_count", 0) >= 4 for r in results),
}, ensure_ascii=False, indent=2))

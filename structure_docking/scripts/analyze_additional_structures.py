import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(r"structure_docking/11_additional_6")
RULES = ROOT / "04_rules" / "extension_rule_audit.json"
MANIFEST = ROOT / "05_structures" / "structure_manifest.json"
OUT = ROOT / "06_structure_analysis"
OUT.mkdir(parents=True, exist_ok=True)
AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}


def parse_pdb(path):
    atoms, residues = [], {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            atom = {"name":line[12:16].strip(),"resname":line[17:20].strip(),"chain":line[21].strip() or "A","resnum":int(line[22:26]),"coord":(float(line[30:38]),float(line[38:46]),float(line[46:54])),"plddt":float(line[60:66])}
        except ValueError:
            continue
        atoms.append(atom); residues.setdefault((atom["chain"],atom["resnum"],atom["resname"]),{})[atom["name"]]=atom
    ordered=sorted(residues,key=lambda x:(x[0],x[1])); sequence="".join(AA3.get(x[2],"X") for x in ordered)
    return atoms,residues,sequence


def atom_for(residues, resnum, preferred):
    for (_,number,_),atoms in residues.items():
        if number==resnum:
            for name in preferred:
                if name in atoms:return atoms[name]
            return atoms.get("CA")
    return None


def triads(residues, nucleophile):
    ns=[]; hs=[]; acids=[]
    for (_,num,res),atoms in residues.items():
        if res==nucleophile:
            name="SG" if res=="CYS" else "OG"
            if name in atoms:ns.append((num,atoms[name]))
        elif res=="HIS":
            hs.extend((num,atoms[name]) for name in ("ND1","NE2") if name in atoms)
        elif res in {"ASP","GLU"}:
            names=("OD1","OD2") if res=="ASP" else ("OE1","OE2")
            acids.extend((num,res,atoms[name]) for name in names if name in atoms)
    found=[]
    for nnum,natom in ns:
        for hnum,hatom in hs:
            nh=math.dist(natom["coord"],hatom["coord"])
            if nh>6:continue
            near=[(math.dist(hatom["coord"],aatom["coord"]),anum,ares,aatom) for anum,ares,aatom in acids if anum not in {nnum,hnum} and math.dist(hatom["coord"],aatom["coord"])<=6]
            if near:
                ha,anum,ares,aatom=min(near,key=lambda x:x[0]); found.append({"nucleophile":f"{nucleophile}{nnum}","his":f"HIS{hnum}","acid":f"{ares}{anum}","nucleophile_his_distance":round(nh,3),"his_acid_distance":round(ha,3),"center":natom["coord"],"plddt":natom["plddt"]})
    return sorted(found,key=lambda x:x["nucleophile_his_distance"]+x["his_acid_distance"])


rules=json.loads(RULES.read_text(encoding="utf-8")); by_rank={int(r["rank"]):r for r in rules["eligible"]}
manifest=json.loads(MANIFEST.read_text(encoding="utf-8")); results=[]
for item in manifest:
    rank=int(item["rank"]); src=by_rank[rank]
    if item["status"]!="downloaded":
        results.append({"rank":rank,"accession":item["accession"],"status":"structure_failed","reason":item.get("reason","")});continue
    atoms,residues,sequence=parse_pdb(Path(item["pdb_file"])); hxxxd=[{"start":m.start()+1,"motif":m.group()} for m in re.finditer(r"H...D",sequence)]; cys=triads(residues,"CYS"); ser=triads(residues,"SER")
    features=src.get("features",""); active=[int(x) for x in re.findall(r"Active site:(\d+)-",features)]; binding=[int(x) for x in re.findall(r"Binding site:(\d+)-",features)]
    acyl=[]
    for match in re.finditer(r"Active site:(\d+)-\d+:Acyl-thioester intermediate",features,re.I):
        pos=int(match.group(1)); atom=atom_for(residues,pos,("SG",))
        if atom and atom["resname"]=="CYS":acyl.append({"residue":f"CYS{pos}","center":atom["coord"],"plddt":atom["plddt"]})
    sites=[]
    for pos in sorted(set(active+binding)):
        atom=atom_for(residues,pos,("SG","OG","ND1","NE2","NZ","CA"))
        if atom:sites.append({"residue":f"{atom['resname']}{pos}","center":atom["coord"],"plddt":atom["plddt"]})
    tm=len(re.findall(r"Transmembrane:",features)); ca=[a for a in atoms if a["name"]=="CA"]
    evidence=[]
    if acyl:evidence.append("UniProt标注酰基硫酯Cys")
    if cys:evidence.append("空间Cys-His-Asp/Glu样口袋")
    if ser:evidence.append("空间Ser-His-Asp/Glu样口袋")
    if hxxxd:evidence.append("HXXXD基序")
    if sites:evidence.append("有注释催化/结合位点")
    if tm>=4:evidence.append(f"多跨膜({tm})")
    priority=0 if acyl else 1 if cys else 2 if hxxxd or ser else 3 if sites else 4
    results.append({"rank":rank,"y_pred":src["y_pred"],"accession":item["accession"],"gene_primary":src.get("gene_primary",""),"protein_name":src.get("protein_name",""),"organism":src.get("organism",""),"status":"analyzed","sequence_length":len(sequence),"mean_plddt":round(sum(a["plddt"] for a in ca)/max(1,len(ca)),2),"transmembrane_count":tm,"hxxxd_count":len(hxxxd),"hxxxd_motifs":hxxxd,"cys_triad_count":len(cys),"cys_triads":cys,"ser_triad_count":len(ser),"ser_triads":ser,"annotated_acyl_cys":acyl,"annotated_sites":sites,"structural_evidence":"；".join(evidence),"structural_priority":priority,"structure_file":item["pdb_file"]})

(OUT/"additional_structure_analysis.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
fields=sorted({k for r in results for k in r})
with (OUT/"additional_structure_analysis.csv").open("w",encoding="utf-8-sig",newline="") as h:
    w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
    for r in results:w.writerow({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v) for k,v in r.items()})
print(json.dumps({"records":len(results),"analyzed":sum(r["status"]=="analyzed" for r in results),"acyl_cys":sum(bool(r.get("annotated_acyl_cys")) for r in results),"cys_triad":sum(r.get("cys_triad_count",0)>0 for r in results),"ser_triad":sum(r.get("ser_triad_count",0)>0 for r in results),"hxxxd":sum(r.get("hxxxd_count",0)>0 for r in results),"multipass":sum(r.get("transmembrane_count",0)>=4 for r in results)},ensure_ascii=False,indent=2))

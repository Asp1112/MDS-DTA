import json
import time
import urllib.request
from pathlib import Path


ROOT = Path(r"E:\total\docking_comparison")
SOURCE = ROOT / "11_additional_6" / "04_rules" / "extension_rule_audit.json"
OUT = ROOT / "11_additional_6" / "05_structures"
OLD_DIRS = [ROOT / "09_corrected_screen" / "structures", ROOT / "07_mechanism_screen" / "structures"]
OUT.mkdir(parents=True, exist_ok=True)


def read_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "MDS-additional-screen/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def download(url, path):
    request = urllib.request.Request(url, headers={"User-Agent": "MDS-additional-screen/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    if b"ATOM" not in payload:
        raise ValueError("not a PDB file")
    path.write_bytes(payload)


source = json.loads(SOURCE.read_text(encoding="utf-8"))
records = []
for candidate in source["eligible"]:
    rank, accession = int(candidate["rank"]), candidate["accession"]
    destination = OUT / f"rank_{rank:04d}_{accession}_AFDB.pdb"
    record = {"rank": rank, "accession": accession, "status": None, "pdb_file": None}
    try:
        if destination.exists():
            record.update(status="downloaded", pdb_file=str(destination), source="existing")
        else:
            reused = next((folder / destination.name for folder in OLD_DIRS if (folder / destination.name).exists()), None)
            if reused:
                destination.write_bytes(reused.read_bytes())
                record.update(status="downloaded", pdb_file=str(destination), source="reused_previous")
            else:
                prediction = read_json(f"https://alphafold.ebi.ac.uk/api/prediction/{accession}")[0]
                download(prediction["pdbUrl"], destination)
                record.update(status="downloaded", pdb_file=str(destination), source="downloaded", pdb_url=prediction["pdbUrl"])
    except Exception as exc:
        record.update(status="failed", reason=str(exc))
    records.append(record)
    (OUT / "structure_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    time.sleep(0.15)

print(json.dumps({"candidates": len(records), "downloaded": sum(r["status"] == "downloaded" for r in records), "failed": [r for r in records if r["status"] == "failed"]}, ensure_ascii=False, indent=2))

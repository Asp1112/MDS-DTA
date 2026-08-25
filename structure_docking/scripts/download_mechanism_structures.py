import json
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(r"E:\total\docking_comparison\07_mechanism_screen")
INPUT = ROOT / "empirical_screen.json"
OUT = ROOT / "structures"
OUT.mkdir(parents=True, exist_ok=True)


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "mechanism-screen/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def download(url, path):
    request = urllib.request.Request(url, headers={"User-Agent": "mechanism-screen/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    if b"ATOM" not in payload:
        raise ValueError("not a PDB file")
    path.write_bytes(payload)


records = []
for candidate in json.loads(INPUT.read_text(encoding="utf-8")):
    if candidate["empirical_disposition"] != "carry_to_structure_screen":
        continue
    rank = int(candidate["rank"])
    accession = candidate["accession"]
    record = {"rank": rank, "accession": accession, "status": None, "pdb_file": None}
    try:
        prediction = get_json(f"https://alphafold.ebi.ac.uk/api/prediction/{accession}")[0]
        pdb_path = OUT / f"rank_{rank:04d}_{accession}_AFDB.pdb"
        if not pdb_path.exists():
            download(prediction["pdbUrl"], pdb_path)
        record.update(
            status="downloaded", pdb_file=str(pdb_path), pdb_url=prediction["pdbUrl"],
            model_version=prediction.get("latestVersion"),
        )
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, IndexError) as exc:
        record.update(status="failed", reason=str(exc))
    records.append(record)
    (OUT / "structure_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    time.sleep(0.2)

print(json.dumps({"downloaded": sum(r["status"] == "downloaded" for r in records), "failed": sum(r["status"] == "failed" for r in records)}, ensure_ascii=False))

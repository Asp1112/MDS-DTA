import json
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(os.environ.get("DOCKING_ROOT", str(Path(__file__).resolve().parents[2])))
INPUT = ROOT / "09_corrected_screen" / "corrected_rule_application.json"
OLD = ROOT / "07_mechanism_screen" / "structures"
OUT = ROOT / "09_corrected_screen" / "structures"
OUT.mkdir(parents=True, exist_ok=True)


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "corrected-structure-screen/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def download(url, path):
    request = urllib.request.Request(url, headers={"User-Agent": "corrected-structure-screen/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    if b"ATOM" not in payload:
        raise ValueError("not a PDB file")
    path.write_bytes(payload)


records = []
source = json.loads(INPUT.read_text(encoding="utf-8"))
for candidate in source["rows"]:
    if candidate["forced_exclude"]:
        continue
    rank = int(candidate["rank"])
    accession = candidate["accession"]
    record = {"rank": rank, "accession": accession, "status": None, "pdb_file": None}
    destination = OUT / f"rank_{rank:04d}_{accession}_AFDB.pdb"
    old_path = OLD / destination.name
    try:
        if destination.exists():
            record.update(status="downloaded", pdb_file=str(destination), source="existing_corrected")
        elif old_path.exists():
            destination.write_bytes(old_path.read_bytes())
            record.update(status="downloaded", pdb_file=str(destination), source="reused_previous")
        else:
            prediction = get_json(f"https://alphafold.ebi.ac.uk/api/prediction/{accession}")[0]
            download(prediction["pdbUrl"], destination)
            record.update(
                status="downloaded", pdb_file=str(destination), source="downloaded",
                pdb_url=prediction["pdbUrl"], model_version=prediction.get("latestVersion"),
            )
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, IndexError) as exc:
        record.update(status="failed", reason=str(exc))
    records.append(record)
    (OUT / "structure_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    time.sleep(0.15)

print(json.dumps({
    "candidates": len(records),
    "downloaded": sum(r["status"] == "downloaded" for r in records),
    "reused": sum(r.get("source") == "reused_previous" for r in records),
    "new": sum(r.get("source") == "downloaded" for r in records),
    "failed": [r for r in records if r["status"] == "failed"],
}, ensure_ascii=False, indent=2))

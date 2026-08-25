import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def get_json(url, timeout=60):
    request = urllib.request.Request(url, headers={"User-Agent": "Codex-batch-docking/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def download(url, destination, timeout=120):
    request = urllib.request.Request(url, headers={"User-Agent": "Codex-batch-docking/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    if b"ATOM" not in payload[:100000]:
        raise ValueError("Downloaded file does not look like a PDB structure")
    destination.write_bytes(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dedup-json",
        default=r"structure_docking/03_deduplicated\deduplication_result.json",
    )
    parser.add_argument(
        "--output-dir",
        default=r"structure_docking/04_alphafold_structures",
    )
    parser.add_argument("--minimum-identity", type=float, default=99.9)
    parser.add_argument("--minimum-query-coverage", type=float, default=95.0)
    parser.add_argument("--target-count", type=int, default=10)
    args = parser.parse_args()

    source = json.loads(Path(args.dedup_json).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    downloaded_count = 0

    for candidate in source["retained"]:
        rank = int(candidate["rank"])
        accession = candidate.get("representative_accession")
        record = {
            "rank": rank,
            "y_pred": candidate.get("y_pred"),
            "accession": accession,
            "identity_pct": candidate.get("identity_pct"),
            "query_coverage_pct": candidate.get("query_coverage_pct"),
            "status": None,
            "reason": None,
            "pdb_file": None,
            "pdb_url": None,
            "model_version": None,
        }
        if downloaded_count >= args.target_count:
            record.update(status="not_selected", reason="lower_rank_after_target_count_reached")
            records.append(record)
            continue
        if not accession or accession.startswith("UPI"):
            record.update(status="excluded", reason="no_canonical_uniprot_accession")
            records.append(record)
            continue
        if (candidate.get("identity_pct") or 0) < args.minimum_identity:
            record.update(status="excluded", reason="uniprot_mapping_below_identity_threshold_for_structure_transfer")
            records.append(record)
            continue
        if (candidate.get("query_coverage_pct") or 0) < args.minimum_query_coverage:
            record.update(status="excluded", reason="uniprot_mapping_below_coverage_threshold_for_structure_transfer")
            records.append(record)
            continue

        try:
            predictions = get_json(f"https://alphafold.ebi.ac.uk/api/prediction/{accession}")
            if not predictions:
                raise ValueError("AlphaFold DB returned no prediction")
            prediction = predictions[0]
            pdb_url = prediction.get("pdbUrl")
            if not pdb_url:
                raise ValueError("AlphaFold DB record has no PDB URL")
            pdb_path = output_dir / f"rank_{rank:04d}_{accession}_AFDB.pdb"
            if not pdb_path.exists():
                download(pdb_url, pdb_path)
            record.update(
                status="downloaded",
                pdb_file=str(pdb_path),
                pdb_url=pdb_url,
                model_version=prediction.get("latestVersion"),
                uniprot_description=prediction.get("uniprotDescription"),
                organism=prediction.get("organismScientificName"),
            )
            downloaded_count += 1
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            record.update(status="excluded", reason=f"alphafold_download_failed: {exc}")
        records.append(record)
        (output_dir / "structure_manifest.json").write_text(
            json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        time.sleep(0.25)

    summary = {
        "downloaded": sum(r["status"] == "downloaded" for r in records),
        "excluded": sum(r["status"] == "excluded" for r in records),
        "not_selected": sum(r["status"] == "not_selected" for r in records),
    }
    payload = {
        "method": {
            "source": "AlphaFold Protein Structure Database API (UniProt-linked AlphaFold2 predictions)",
            "minimum_uniprot_mapping_identity_pct": args.minimum_identity,
            "minimum_query_coverage_pct": args.minimum_query_coverage,
        },
        "summary": summary,
        "records": records,
    }
    (output_dir / "structure_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

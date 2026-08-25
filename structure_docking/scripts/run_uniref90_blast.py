import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast"


def get_text(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "MDS-additional-screen/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8").strip()


def post_form(url, data, timeout=90):
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=encoded, headers={"User-Agent": "MDS-additional-screen/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-rank", type=int, required=True)
    parser.add_argument("--end-rank", type=int, required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "jobs.json"
    source = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    records = {int(r["rank"]): r for r in source["records"] if args.start_rank <= int(r["rank"]) <= args.end_rank}
    jobs = {}
    if state_path.exists():
        try:
            saved = json.loads(state_path.read_text(encoding="utf-8-sig"))
            jobs = {int(j["rank"]): j for j in saved.get("jobs", []) if j.get("job_id")}
        except (ValueError, TypeError):
            jobs = {}

    def save():
        state_path.write_text(json.dumps({"service": "EMBL-EBI NCBI BLAST REST", "database": "uniref90", "jobs": [jobs[k] for k in sorted(jobs)]}, ensure_ascii=False, indent=2), encoding="utf-8")

    for rank in sorted(records):
        if rank in jobs:
            continue
        record = records[rank]
        try:
            job_id = post_form(f"{BASE}/run", {
                "email": args.email, "title": f"MDS_rank_{rank:04d}", "program": "blastp", "database": "uniref90",
                "sequence": record["protein_sequence"], "stype": "protein", "alignments": "50", "scores": "50",
                "exp": "1e-5", "matrix": "BLOSUM62", "filter": "F",
            })
            jobs[rank] = {"rank": rank, "score": record["y_pred"], "job_id": job_id, "status": "SUBMITTED", "result_file": None, "last_error": None}
            save()
            print(f"submitted {rank} {job_id}", flush=True)
            time.sleep(0.25)
        except Exception as exc:
            print(f"submission failed {rank}: {exc}", flush=True)
            time.sleep(3)

    def check(item):
        rank, job = item
        try:
            status = get_text(f"{BASE}/status/{job['job_id']}", timeout=45)
            return rank, status, None
        except Exception as exc:
            return rank, None, str(exc)

    while True:
        pending = [(rank, job) for rank, job in jobs.items() if job.get("status") not in {"FINISHED", "ERROR", "FAILURE", "NOT_FOUND"} or (job.get("status") == "FINISHED" and not job.get("result_file"))]
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=15) as pool:
            futures = [pool.submit(check, item) for item in pending]
            for future in as_completed(futures):
                rank, status, error = future.result()
                if error:
                    jobs[rank]["last_error"] = error
                    continue
                jobs[rank]["status"] = status
                if status == "FINISHED" and not jobs[rank].get("result_file"):
                    path = output / f"rank_{rank:04d}.xml"
                    try:
                        text = get_text(f"{BASE}/result/{jobs[rank]['job_id']}/xml", timeout=90)
                        path.write_text(text, encoding="utf-8")
                        jobs[rank]["result_file"] = str(path)
                    except Exception as exc:
                        jobs[rank]["last_error"] = str(exc)
        save()
        counts = {}
        for job in jobs.values():
            counts[job.get("status", "UNKNOWN")] = counts.get(job.get("status", "UNKNOWN"), 0) + 1
        downloaded = sum(bool(job.get("result_file")) for job in jobs.values())
        print(json.dumps({"status": counts, "downloaded": downloaded}, ensure_ascii=False), flush=True)
        if downloaded == len(records):
            break
        time.sleep(10)

    save()
    print(json.dumps({"submitted": len(jobs), "downloaded": sum(bool(j.get("result_file")) for j in jobs.values())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def first_descendant(node, local_name):
    return node.find(f".//{{*}}{local_name}")


def text_float(node, local_name, default=0.0):
    child = first_descendant(node, local_name)
    return float(child.text) if child is not None and child.text else default


def parse_blast_xml(path):
    root = ET.parse(path).getroot()
    query = first_descendant(root, "sequence")
    hit = first_descendant(root, "hit")
    if hit is None:
        return {
            "hit_found": False,
            "uniref90_cluster": None,
            "representative_accession": None,
            "identity_pct": None,
            "query_coverage_pct": None,
            "hit_coverage_pct": None,
        }

    alignment = first_descendant(hit, "alignment")
    query_seq = first_descendant(alignment, "querySeq")
    match_seq = first_descendant(alignment, "matchSeq")
    query_length = int(query.attrib.get("length", 0)) if query is not None else 0
    hit_length = int(hit.attrib.get("length", 0))
    query_span = int(query_seq.attrib["end"]) - int(query_seq.attrib["start"]) + 1
    hit_span = int(match_seq.attrib["end"]) - int(match_seq.attrib["start"]) + 1
    cluster = hit.attrib.get("ac") or hit.attrib.get("id")
    representative = re.sub(r"^UniRef90_", "", cluster or "") or None

    return {
        "hit_found": True,
        "uniref90_cluster": cluster,
        "representative_accession": representative,
        "hit_description": hit.attrib.get("description"),
        "identity_pct": text_float(alignment, "identity"),
        "query_coverage_pct": round(100.0 * query_span / query_length, 3) if query_length else None,
        "hit_coverage_pct": round(100.0 * hit_span / hit_length, 3) if hit_length else None,
        "expectation": text_float(alignment, "expectation"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", default=r"E:\total\docking_comparison\01_rank_pool\rank_1_200_for_uniprot.json")
    parser.add_argument("--blast-dir", default=r"E:\total\docking_comparison\02_uniref90_blast")
    parser.add_argument("--output-dir", default=r"E:\total\docking_comparison\03_deduplicated")
    parser.add_argument("--reference-end", type=int, default=100)
    parser.add_argument("--candidate-start", type=int, default=101)
    parser.add_argument("--candidate-end", type=int, default=200)
    parser.add_argument("--identity-threshold", type=float, default=90.0)
    parser.add_argument("--coverage-threshold", type=float, default=80.0)
    args = parser.parse_args()

    source = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    records_by_rank = {int(r["rank"]): dict(r) for r in source["records"]}
    blast_dir = Path(args.blast_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed = {}
    missing = []
    for rank in range(1, args.candidate_end + 1):
        xml_path = blast_dir / f"rank_{rank:04d}.xml"
        if not xml_path.exists():
            missing.append(rank)
            continue
        parsed[rank] = parse_blast_xml(xml_path)

    if missing:
        raise SystemExit(f"Missing BLAST XML files for ranks: {missing[:20]}{'...' if len(missing) > 20 else ''}")

    for rank, blast in parsed.items():
        blast["mapping_pass"] = bool(
            blast["hit_found"]
            and blast["identity_pct"] >= args.identity_threshold
            and blast["query_coverage_pct"] >= args.coverage_threshold
        )
        records_by_rank[rank].update(blast)

    reference_clusters = {}
    reference_sequences = {}
    for rank in range(1, args.reference_end + 1):
        record = records_by_rank[rank]
        reference_sequences.setdefault(record["protein_sequence"], rank)
        if record["mapping_pass"]:
            reference_clusters.setdefault(record["uniref90_cluster"], rank)

    retained = []
    retained_clusters = {}
    retained_sequences = {}
    evaluated = []
    for rank in range(args.candidate_start, args.candidate_end + 1):
        record = records_by_rank[rank]
        reason = None
        redundant_with_rank = None

        if record["protein_sequence"] in reference_sequences:
            reason = "exact_duplicate_of_rank_1_100"
            redundant_with_rank = reference_sequences[record["protein_sequence"]]
        elif record["protein_sequence"] in retained_sequences:
            reason = "exact_duplicate_within_candidate_block"
            redundant_with_rank = retained_sequences[record["protein_sequence"]]
        elif record["mapping_pass"] and record["uniref90_cluster"] in reference_clusters:
            reason = "same_uniref90_cluster_as_rank_1_100"
            redundant_with_rank = reference_clusters[record["uniref90_cluster"]]
        elif record["mapping_pass"] and record["uniref90_cluster"] in retained_clusters:
            reason = "same_uniref90_cluster_as_higher_candidate"
            redundant_with_rank = retained_clusters[record["uniref90_cluster"]]

        selected = reason is None
        record["dedup_selected"] = selected
        record["dedup_exclusion_reason"] = reason
        record["redundant_with_rank"] = redundant_with_rank
        evaluated.append(record)

        if selected:
            retained.append(record)
            retained_sequences[record["protein_sequence"]] = rank
            if record["mapping_pass"]:
                retained_clusters[record["uniref90_cluster"]] = rank

    next_block = None if len(retained) >= 10 else {
        "start": args.candidate_end + 1,
        "end": args.candidate_end + 100,
    }
    result = {
        "method": {
            "database": "UniRef90 via EMBL-EBI NCBI BLAST REST",
            "identity_threshold_pct": args.identity_threshold,
            "minimum_query_coverage_pct": args.coverage_threshold,
            "ranking_rule": "greedy ascending rank; retain the highest-ranked sequence",
            "reference_ranks": [1, args.reference_end],
            "candidate_ranks": [args.candidate_start, args.candidate_end],
            "extension_rule": "if fewer than 10 candidates survive all downstream filters, add the next complete block of 100 ranks",
        },
        "summary": {
            "evaluated_count": len(evaluated),
            "dedup_retained_count": len(retained),
            "dedup_excluded_count": len(evaluated) - len(retained),
            "next_block_if_needed": next_block,
        },
        "retained": retained,
        "evaluated": evaluated,
    }
    (output_dir / "deduplication_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "deduplication_result.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "rank", "y_pred", "sequence_length", "uniref90_cluster", "representative_accession",
            "identity_pct", "query_coverage_pct", "hit_coverage_pct", "mapping_pass",
            "dedup_selected", "dedup_exclusion_reason", "redundant_with_rank", "protein_sequence",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in evaluated:
            record["sequence_length"] = len(record["protein_sequence"])
            writer.writerow(record)

    fasta = "\n".join(
        f">rank_{r['rank']:04d}|score={r['y_pred']:.9f}|{r.get('uniref90_cluster') or 'no_cluster'}\n{r['protein_sequence']}"
        for r in retained
    )
    (output_dir / "deduplicated_candidates.fasta").write_text(fasta + ("\n" if fasta else ""), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

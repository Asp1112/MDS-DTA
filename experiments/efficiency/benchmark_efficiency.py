"""Measure reproducible MDS inference efficiency on a processed dataset.

This script reports trainable parameters, latency per protein-ligand pair,
throughput, and peak CUDA memory. FLOPs for PyTorch-Geometric message passing
are not reported because common profilers do not cover every sparse operator;
if FLOPs are added, the profiler coverage and representative input sizes must
be stated explicitly in the manuscript.
"""

import argparse
import json
import statistics
import time

import torch
from torch_geometric.data import DataLoader

from models.MDS_DTA import MDSDTA
from utils import TestbedDataset


def load_model(checkpoint, device):
    payload = torch.load(checkpoint, map_location=device)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    state = {key.replace("module.", ""): value for key, value in state.items()}
    model = MDSDTA().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Processed dataset name, e.g. bindingdb_test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--output", default="efficiency_metrics.json")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    dataset = TestbedDataset(root="data", dataset=args.dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader)).to(device)
    model = load_model(args.checkpoint, device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(batch)
        synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(args.repetitions):
            start = time.perf_counter()
            model(batch)
            synchronize(device)
            times.append(time.perf_counter() - start)

    pairs = int(batch.num_graphs)
    mean_batch_s = statistics.mean(times)
    result = {
        "device": str(device),
        "dataset": args.dataset,
        "batch_size": pairs,
        "warmup_iterations": args.warmup,
        "timed_repetitions": args.repetitions,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "mean_batch_latency_ms": mean_batch_s * 1000,
        "sd_batch_latency_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0.0,
        "mean_latency_ms_per_pair": mean_batch_s * 1000 / pairs,
        "throughput_pairs_per_second": pairs / mean_batch_s,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None
        ),
        "flops": None,
        "flops_note": "Not reported: sparse PyG operator coverage must be validated before use.",
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

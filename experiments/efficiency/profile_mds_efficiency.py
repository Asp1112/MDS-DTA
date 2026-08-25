import argparse, csv, json, os, platform, subprocess, time
from pathlib import Path
import numpy as np
import torch
from train_test import averaged_state, build_model, make_loaders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="davis")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=40)
    ap.add_argument("--out", default="efficiency_cost_davis.json")
    args = ap.parse_args()
    run = Path(args.run_dir)
    cfg = json.loads((run / "config.json").read_text())
    model, _, _, applied, _ = build_model(cfg["model_module"].split("models.")[-1], json.dumps(cfg.get("model_parameters", {})))
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    checkpoints = []
    for p in sorted(run.glob("checkpoint_epoch*_mse*.pt")):
        obj = torch.load(p, map_location="cpu", weights_only=True)
        checkpoints.append({"epoch": obj.get("epoch", 0), "mse": obj.get("mse", 0), "path": p})
    if not checkpoints:
        raise SystemExit("No checkpoint files in run directory")
    model.load_state_dict(averaged_state(checkpoints[:3]))
    split = Path("splits") / args.dataset / f"fold_{args.fold}.json"
    _, _, test_loader, sizes, _ = make_loaders(device, args.dataset, split, args.batch_size, args.batch_size)
    batch = next(iter(test_loader)).to(device)
    pairs = int(batch.y.numel())
    def forward():
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
            return model(batch)
    for _ in range(args.warmup): forward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.repeats):
        a = torch.cuda.Event(True); b = torch.cuda.Event(True)
        a.record(); forward(); b.record(); torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    median_batch_ms = float(np.median(times))
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, with_flops=True) as prof:
        forward(); torch.cuda.synchronize()
    flops_batch = float(sum((e.flops or 0) for e in prof.key_averages()))
    hist = list(csv.DictReader((run / "history.csv").open()))
    seconds = [float(r["seconds"]) for r in hist if r.get("seconds")]
    total_s = float(sum(seconds))
    ckpt_size_mb = sum(p.stat().st_size for p in run.glob("checkpoint_epoch*_mse*.pt")) / max(1, len(list(run.glob("checkpoint_epoch*_mse*.pt")))) / 1024**2
    cpu = subprocess.check_output("lscpu | grep 'Model name' | head -1 | cut -d: -f2-", shell=True, text=True).strip()
    gpu = torch.cuda.get_device_name(0)
    result = {
        "dataset": args.dataset, "fold": args.fold, "run_dir": str(run),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "flops_per_pair_profiler_estimate": flops_batch / pairs,
        "macs_per_pair_estimate": flops_batch / pairs / 2,
        "model_file_size_mb": ckpt_size_mb,
        "training_time_per_epoch_s_mean": float(np.mean(seconds)),
        "training_time_per_epoch_s_std": float(np.std(seconds)),
        "epochs_completed": len(hist),
        "total_training_time_h": total_s / 3600,
        "inference_latency_ms_per_pair": median_batch_ms / pairs,
        "inference_throughput_pairs_s": pairs / (median_batch_ms / 1000),
        "peak_gpu_memory_gb_inference_batch": peak_gb,
        "gpu_model": gpu, "cpu_model": cpu,
        "batch_size": cfg.get("batch_size", args.batch_size),
        "inference_batch_size": pairs,
        "software_environment": {"python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
        "notes": "FLOPs are torch.profiler estimates for supported operators; MACs=FLOPs/2. Latency is median CUDA-event time after warm-up. Training fields come from the completed Davis fold record."
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

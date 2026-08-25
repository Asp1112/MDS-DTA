"""Run the three CombinedDTA fusion variants sequentially on Davis.

Each variant is trained with the formal Davis protocol capped at 300 epochs
so the three models can be compared at the same training budget:

    CombinedDTA V2  ->  V2-B  ->  V2-C

Every run writes its own directory under results/ (config.json, history.csv,
validation_summary.json, test_metrics.json, test_predictions.npz). After all
three finish, this script writes results/comparison_summary.json with the
test metrics side by side.

Usage:
  python run_all_variants.py [--epochs 300]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
MODELS = ["v2", "b", "c"]


def latest_run_dir(class_name):
    prefix = class_name + "_davis_"
    candidates = [
        d for d in (ROOT / "results").glob("*_davis_*")
        if d.is_dir() and d.name.startswith(prefix)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.name)


def main():
    parser = argparse.ArgumentParser(description="Run the three variants sequentially on Davis.")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Maximum epochs per model (default 300).")
    args = parser.parse_args()

    summary = {
        "epochs": args.epochs,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
    }
    print(f"Python: {PYTHON}", flush=True)
    print(f"Will run {len(MODELS)} models sequentially, {args.epochs} epochs each.", flush=True)

    for model in MODELS:
        print(f"\n===== [{time.strftime('%H:%M:%S')}] Starting model '{model}' "
              f"({args.epochs} epochs) =====", flush=True)
        t0 = time.time()
        proc = subprocess.run(
            [PYTHON, str(ROOT / "train_mds_variants.py"),
             "--model", model, "--epochs", str(args.epochs)],
            cwd=str(ROOT))
        elapsed_min = (time.time() - t0) / 60.0
        print(f"===== [{time.strftime('%H:%M:%S')}] Model '{model}' finished "
              f"(exit={proc.returncode}, {elapsed_min:.1f} min) =====", flush=True)

        entry = {"exit_code": proc.returncode, "elapsed_minutes": round(elapsed_min, 1)}
        if proc.returncode == 0:
            class_name = {"v2": "CombinedDTAV2",
                          "b": "CombinedDTAV2B",
                          "c": "CombinedDTAV2C"}[model]
            run_dir = latest_run_dir(class_name)
            if run_dir is not None:
                entry["run_dir"] = str(run_dir)
                test_path = run_dir / "test_metrics.json"
                val_path = run_dir / "validation_summary.json"
                if test_path.exists():
                    entry["test_metrics"] = json.loads(test_path.read_text(encoding="utf-8"))
                if val_path.exists():
                    entry["validation"] = json.loads(val_path.read_text(encoding="utf-8"))
        summary["models"][model] = entry

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path = ROOT / "results" / "comparison_summary.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nComparison summary written to {out_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

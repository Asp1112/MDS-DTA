#!/usr/bin/env python3
"""run_sixfold_cv.py - six-fold rotation for the CombinedDTA-family formal runs.

Protocol (as encoded in splits/<dataset>/fold_<N>.json by create_data.py):
for outer fold N, fold N is the one-time TEST set, fold (N+1) mod 6 is the
VALIDATION set, and the other four folds are TRAINING data.  The fixed
4/1/1 run that is currently executing is exactly fold 0, so by default this
script runs the other five folds (1..5) for each requested dataset.

Runs execute sequentially in the foreground so you can watch the training
process live; every run is also teed to sixfold_logs/<ds>_fold<N>_*.log.
A fold is skipped if a completed run already exists under
results/cv/<dataset>/.

By default the script waits for any other training job on the GPU (for
example the currently running combined_dta_token experiment) to finish
before starting.

Examples:
  python run_sixfold_cv.py --model combined_dta_token            # davis, folds 1-5
  python run_sixfold_cv.py --model combined_dta_token --dataset davis --folds 1 2 3 4 5
  python run_sixfold_cv.py --model combined_dta_control --entry train_test.py
  python run_sixfold_cv.py --model combined_dta_lstmdrop --entry train.py
  python run_sixfold_cv.py --model combined_dta_token --all-folds
  python run_sixfold_cv.py --model combined_dta_token --start-now
  python run_sixfold_cv.py --model combined_dta_token --dry-run
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time


ROOT = "/root/mds"
PYTHON = "/root/mds/.venv_dta/bin/python"
LOG_DIR = os.path.join(ROOT, "sixfold_logs")
STATUS_FILE = os.path.join(ROOT, "sixfold_status.json")
LOCK_FILE = os.path.join(ROOT, ".sixfold.lock")


def write_status(phase, detail, pid=""):
    status = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": phase,
        "detail": detail,
        "pid": str(pid),
    }
    with open(STATUS_FILE, "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)
    print(phase, "-", detail, flush=True)


def other_gpu_job_running():
    proc = subprocess.run(
        ["pgrep", "-f", r"train_test\.py|train\.py|extract_esm_features\.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def wait_for_gpu(start_now):
    if start_now:
        print("Skipping the GPU wait (--start-now).", flush=True)
        return
    while other_gpu_job_running():
        print("[%s] Another training job is using the GPU; waiting 60 s ..."
              % time.strftime("%H:%M:%S"), flush=True)
        write_status("waiting_for_gpu", "another training job is running")
        time.sleep(60)
    print("GPU is free; starting the next fold.", flush=True)


def dataset_done(dataset, fold, model):
    pattern = os.path.join(ROOT, "results", "cv", dataset, model,
                           "*_fold%d_*" % fold, "test_metrics.json")
    return bool(glob.glob(pattern))


def run_fold(dataset, fold, model, entry, model_params):
    out_root = os.path.join("results", "cv", dataset, model)
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    if dataset_done(dataset, fold, model):
        print("[skip] %s fold %d: already completed under results/cv/%s/%s/"
              % (dataset, fold, dataset, model), flush=True)
        write_status("skipped_%s_%s_fold%d" % (model, dataset, fold),
                     "already completed")
        return True

    log_path = os.path.join(LOG_DIR, "%s_fold%d_%s.log"
                            % (dataset, fold, time.strftime("%Y%m%d-%H%M%S")))
    print("=== [%s fold %d] %s - starting (log: %s) ==="
          % (dataset, fold, time.strftime("%Y-%m-%d %H:%M:%S"), log_path),
          flush=True)
    write_status("training_%s_fold%d" % (dataset, fold),
                 "model=%s, log: %s" % (model, log_path))

    command = [PYTHON, entry, "--dataset", dataset, "--model", model]
    if model_params:
        command += ["--model-params", model_params]
    command += ["--test-fold", str(fold), "--results-root", out_root]

    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
        rc = proc.wait()

    if rc != 0:
        print("[FAILED] %s fold %d: exit code %d (see %s)"
              % (dataset, fold, rc, log_path), flush=True)
        write_status("failed_%s_fold%d" % (dataset, fold),
                     "exit code %d, log: %s" % (rc, log_path))
        sys.exit(rc)
    print("[OK] %s fold %d finished at %s"
          % (dataset, fold, time.strftime("%Y-%m-%d %H:%M:%S")), flush=True)
    write_status("finished_%s_fold%d" % (dataset, fold), "log: %s" % log_path)
    return True


def write_summary(dataset, model):
    root = os.path.join(ROOT, "results", "cv", dataset, model)
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*_fold*_*", "test_metrics.json"))):
        run_dir = os.path.dirname(path)
        match = re.search(r"_fold(\d+)_", os.path.basename(run_dir))
        if not match:
            continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        rows.append({
            "fold": int(match.group(1)),
            "run_dir": os.path.relpath(run_dir, ROOT),
            "test_metrics": data,
        })
    rows.sort(key=lambda r: r["fold"])
    summary = {"dataset": dataset, "n_folds": len(rows),
               "folds": rows, "aggregate": {}}
    if rows:
        def aggregate(prefix, key):
            values = [r["test_metrics"][prefix].get(key) for r in rows]
            values = [v for v in values if v is not None]
            if not values:
                return None
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            return {"mean": mean, "std": math.sqrt(var), "values": values}
        for prefix in ("best_checkpoint", "top3_average"):
            for key in ("mse", "rmse", "pearson", "spearman",
                        "ci", "r2", "rm2", "mae"):
                summary["aggregate"]["%s_%s" % (prefix, key)] = aggregate(prefix, key)
    out = os.path.join(ROOT, "results", "cv",
                       dataset + "_" + model + "_sixfold_summary.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("summary written:", out, flush=True)


def preflight(datasets, folds, model, entry):
    if not os.path.isfile(os.path.join(ROOT, entry)):
        print("Training entry not found: %s" % entry)
        sys.exit(1)
    probe = ("import importlib,sys; "
             "mod='models.%s'; "
             "importlib.import_module(mod)" % model)
    check = subprocess.run([PYTHON, "-c", probe], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if check.returncode != 0:
        probe = "import importlib; importlib.import_module('%s')" % model
        check = subprocess.run([PYTHON, "-c", probe], cwd=ROOT,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if check.returncode != 0:
        print("Model module not importable: models.%s" % model)
        sys.exit(1)
    for dataset in datasets:
        for fold in folds:
            if not os.path.isfile(os.path.join(ROOT, "splits", dataset,
                                               "fold_%d.json" % fold)):
                print("Missing split file: splits/%s/fold_%d.json" % (dataset, fold))
                sys.exit(1)
        if not os.path.exists(os.path.join(
                ROOT, "data", "processed", dataset + "_sixfold_all.pt")):
            print("Missing data file: data/processed/%s_sixfold_all.pt "
                  "(run prepare_sixfold_data.py first)." % dataset)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Six-fold rotation: run davis (or other datasets) with the "
                    "other five folds as test sets, one at a time.")
    parser.add_argument("--datasets", "--dataset", nargs="*", default=["davis"],
                        help="Dataset name(s), default: davis.")
    parser.add_argument("--folds", nargs="*", type=int, default=[1, 2, 3, 4, 5],
                        help="Fold ids to run (0..5). Default 1 2 3 4 5, i.e. "
                             "all folds except the currently running fixed "
                             "4/1/1 split (fold 0).")
    parser.add_argument("--all-folds", action="store_true",
                        help="Run folds 0 1 2 3 4 5 (fold 0 duplicates the "
                             "already-running fixed split).")
    parser.add_argument("--model", required=True,
                        help="Model module name under models/, e.g. "
                             "combined_dta_token. Fill in the model you want "
                             "to evaluate.")
    parser.add_argument("--model-params", default=None,
                        help="Optional JSON object overriding model "
                             "hyper-parameters.")
    parser.add_argument("--entry", default="train_test.py",
                        help="Training entry script (default: train_test.py).")
    parser.add_argument("--start-now", action="store_true",
                        help="Do not wait for the currently running GPU job.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit without training.")
    args = parser.parse_args()

    if args.all_folds:
        args.folds = [0, 1, 2, 3, 4, 5]
    args.folds = sorted(set(args.folds))
    for fold in args.folds:
        if fold not in range(6):
            print("Invalid fold id: %d (must be 0..5)." % fold)
            sys.exit(1)
    if not args.folds:
        print("No folds selected.")
        sys.exit(1)

    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE, encoding="utf-8") as fh:
            print("Another six-fold run is already active (%s). Exiting."
                  % fh.read().strip())
        sys.exit(1)
    with open(LOCK_FILE, "w", encoding="utf-8") as fh:
        fh.write("pid %d started %s"
                 % (os.getpid(), time.strftime("%Y-%m-%d %H:%M:%S")))
    try:
        print("=== Six-fold rotation (dataset(s): %s | folds: %s | "
              "entry: %s | model: %s%s) ==="
              % (", ".join(args.datasets), " ".join(map(str, args.folds)),
                 args.entry, args.model,
                 (" | params: " + args.model_params) if args.model_params else ""))
        print("Start: %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)

        preflight(args.datasets, args.folds, args.model, args.entry)

        if args.dry_run:
            print("DRY RUN - planned jobs:")
            for dataset in args.datasets:
                for fold in args.folds:
                    if dataset_done(dataset, fold, args.model):
                        print("  [skip] %s fold %d (already completed)"
                              % (dataset, fold))
                    else:
                        print("  [run ] %s fold %d -> results/cv/%s/%s/"
                              % (dataset, fold, dataset, args.model))
            sys.exit(0)

        wait_for_gpu(args.start_now)

        for dataset in args.datasets:
            print("\n########## Dataset: %s ##########" % dataset, flush=True)
            for fold in args.folds:
                run_fold(dataset, fold, args.model, args.entry, args.model_params)
            write_summary(dataset, args.model)

        write_status("done", "all requested folds completed")
        print("\nAll done: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
        print("Per-model summaries: results/cv/<dataset>_<model>_sixfold_summary.json")
    finally:
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass


if __name__ == "__main__":
    main()

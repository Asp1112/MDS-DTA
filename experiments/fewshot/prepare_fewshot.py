"""Build the few-shot (reduced-data) datasets on the fixed six-part division.

For every outer fold N the sample-level six-fold protocol is reused:
test = fold N, validation = fold (N+1) mod 6, and the full training pool is
the other four folds.  A fraction of that training pool (50%, 25% or 10%) is
then drawn without replacement with a fixed seed and stored as the
``train_indices`` of the manifest; validation and test are never reduced.

Usage (from this folder):
  python prepare_fewshot.py --datasets davis kiba bindingdb
"""

import argparse
import json
import os
import random
from pathlib import Path


MDS_ROOT = Path(os.environ.get("MDS_ROOT", "/root/mds")).resolve()
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = HERE / "data"
FRACTIONS = [0.5, 0.25, 0.1]


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def frac_tag(frac):
    return "fs%d" % int(round(frac * 100))


def sample_fold_ids(dataset):
    path = MDS_ROOT / "splits" / dataset / "fold_membership.json"
    with path.open(encoding="utf-8") as fh:
        membership = json.load(fh)
    fold_members = membership["fold_members"]
    n = membership["n_samples"]
    fold_ids = [None] * n
    for fold, members in enumerate(fold_members):
        for i in members:
            fold_ids[i] = fold
    if any(f is None for f in fold_ids):
        raise SystemExit(f"fold_membership does not cover all {n} samples: {path}")
    return fold_ids, n


def prepare_dataset(dataset, seed):
    fold_ids, n = sample_fold_ids(dataset)
    rng = random.Random(seed)
    summaries = {}
    for frac in FRACTIONS:
        tag = frac_tag(frac)
        setting_dir = OUTPUT_ROOT / "splits" / dataset / tag
        ensure_dir(setting_dir)
        fold_summaries = []
        for fold in range(6):
            val_fold = (fold + 1) % 6
            test = [i for i in range(n) if fold_ids[i] == fold]
            validation = [i for i in range(n) if fold_ids[i] == val_fold]
            train_pool = [i for i in range(n)
                          if fold_ids[i] not in (fold, val_fold)]
            train_pool.sort()
            n_sub = int(round(frac * len(train_pool)))
            subset = sorted(rng.sample(train_pool, n_sub))
            manifest = {
                "schema_version": 1,
                "dataset": dataset,
                "experiment": "fewshot",
                "fraction": frac,
                "setting": tag,
                "outer_fold": fold,
                "seed": seed,
                "protocol": (
                    f"sample-level six-fold rotation: test=fold {fold}, "
                    f"validation=fold {val_fold}, train={frac:.0%} of the "
                    f"other four folds ({n_sub}/{len(train_pool)})"),
                "train_indices": subset,
                "validation_indices": validation,
                "test_indices": test,
                "sizes": {"train": len(subset),
                          "validation": len(validation), "test": len(test),
                          "train_pool": len(train_pool)},
                "audit": {
                    "subset_of_train_pool": set(subset).issubset(set(train_pool)),
                    "train_fraction": round(n_sub / max(1, len(train_pool)), 6),
                },
            }
            out = setting_dir / f"fold_{fold}.json"
            with out.open("w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=1)
            fold_summaries.append({
                "fold": fold, "sizes": manifest["sizes"],
                "audit": manifest["audit"],
            })
            print(f"{dataset} {tag} fold {fold}: "
                  f"train={len(subset)}/{len(train_pool)} val={len(validation)} "
                  f"test={len(test)}", flush=True)
        summaries[tag] = fold_summaries
    summary_dir = OUTPUT_ROOT / dataset
    ensure_dir(summary_dir)
    with (summary_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "dataset": dataset, "experiment": "fewshot", "seed": seed,
            "n_samples": n, "fractions": summaries,
        }, fh, indent=2, ensure_ascii=False)
    print(f"[OK] {dataset}: manifests written under "
          f"{OUTPUT_ROOT / 'splits' / dataset}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build six-fold few-shot datasets (50% / 25% / 10%).")
    parser.add_argument("--datasets", nargs="+",
                        default=["davis", "kiba", "bindingdb"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ensure_dir(OUTPUT_ROOT)
    for dataset in args.datasets:
        prepare_dataset(dataset.lower(), args.seed)
    print("Done.")


if __name__ == "__main__":
    main()

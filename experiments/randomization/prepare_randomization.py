"""Build the randomization datasets on the fixed six-part division.

For every outer fold N the sample-level six-fold protocol is reused:
test = fold N, validation = fold (N+1) mod 6, train = the other four folds.
On top of that, the training samples receive a reproducible permutation
(stored as ``train_permutation``, aligned position-wise with
``train_indices``) that the trainer applies to:
  * rand_x1 : swap the protein (target) feature of every training sample;
  * rand_x2 : swap the compound graph (x / edge_index) of every training
              sample;
  * rand_y  : swap the affinity label of every training sample.
The validation and test sets are left untouched so the evaluation remains
meaningful; the randomization destroys the train-side drug-target-label
association only.

Usage (from this folder):
  python prepare_randomization.py --datasets davis kiba bindingdb
"""

import argparse
import json
import os
import random
from pathlib import Path


MDS_ROOT = Path(os.environ.get("MDS_ROOT", "/root/mds")).resolve()
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = HERE / "data"
MODES = ["x1", "x2", "y"]


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


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
    for mode in MODES:
        mode_dir = OUTPUT_ROOT / "splits" / dataset / ("rand_" + mode)
        ensure_dir(mode_dir)
        fold_summaries = []
        for fold in range(6):
            val_fold = (fold + 1) % 6
            test = [i for i in range(n) if fold_ids[i] == fold]
            validation = [i for i in range(n) if fold_ids[i] == val_fold]
            train_pool = [i for i in range(n)
                          if fold_ids[i] not in (fold, val_fold)]
            train_pool.sort()
            permutation = list(train_pool)
            rng.shuffle(permutation)
            shuffled_differ = sum(
                1 for a, b in zip(train_pool, permutation) if a != b)
            manifest = {
                "schema_version": 1,
                "dataset": dataset,
                "experiment": "randomization",
                "mode": "rand_" + mode,
                "outer_fold": fold,
                "seed": seed,
                "protocol": (
                    f"sample-level six-fold rotation: test=fold {fold}, "
                    f"validation=fold {val_fold}, train=other four folds; "
                    f"train-side {mode} permutation applied"),
                "train_indices": train_pool,
                "validation_indices": validation,
                "test_indices": test,
                "train_permutation": permutation,
                "sizes": {"train": len(train_pool),
                          "validation": len(validation), "test": len(test)},
                "audit": {
                    "permutation_is_valid": sorted(permutation) == train_pool,
                    "train_positions_changed": shuffled_differ,
                    "validation_test_untouched": True,
                },
            }
            out = mode_dir / f"fold_{fold}.json"
            with out.open("w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=1)
            fold_summaries.append({
                "fold": fold, "sizes": manifest["sizes"],
                "audit": manifest["audit"],
            })
            print(f"{dataset} rand_{mode} fold {fold}: "
                  f"train={len(train_pool)} val={len(validation)} "
                  f"test={len(test)} positions_changed={shuffled_differ}",
                  flush=True)
        summaries["rand_" + mode] = fold_summaries

    summary_dir = OUTPUT_ROOT / dataset
    ensure_dir(summary_dir)
    with (summary_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "dataset": dataset, "experiment": "randomization", "seed": seed,
            "n_samples": n, "modes": summaries,
        }, fh, indent=2, ensure_ascii=False)
    print(f"[OK] {dataset}: manifests written under "
          f"{OUTPUT_ROOT / 'splits' / dataset}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build six-fold randomization datasets (x1 / x2 / y).")
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

"""Build the cold-start datasets on top of the fixed six-part division.

Protocol (entity-level six-fold):
  * cold_drug  : unique drugs are split into 6 disjoint groups.  Fold N tests
                 every interaction of group-N drugs, folds (N+1) mod 6 is the
                 validation group, and the other four groups train.  Test
                 drugs are therefore completely unseen during training.
  * cold_target: same construction on unique target proteins.
  * cold_both  : unique drugs AND unique targets are each split into 6
                 groups.  Fold N tests pairs with drug group N and target
                 group N, fold (N+1) mod 6 validates pairs from groups N+1,
                 and training uses pairs whose drug and target both come from
                 the other four groups.  Neither a test drug nor a test
                 target can appear in the training pairs.

Every manifest stores the entity-group maps and the train / validation /
test sample indices over the canonical <dataset>_sixfold_all.csv index space,
so the same manifests drive the MDS-family trainer and the DeepDTA / GraphDTA
/ DeepDTAGen cold-start entries.

Usage (from this folder):
  python prepare_cold_start.py --datasets davis kiba bindingdb
"""

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd


MDS_ROOT = Path(os.environ.get("MDS_ROOT", "/root/mds")).resolve()
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = HERE / "data"
SETTINGS = ["cold_drug", "cold_target", "cold_both"]


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def group_entities(rng, entities):
    """Deterministically split unique entities into 6 balanced groups."""
    order = list(entities)
    rng.shuffle(order)
    groups = {}
    for i, entity in enumerate(order):
        groups[entity] = i % 6
    return groups


def fold_partition(dataset, drugs, targets, groups, setting, fold):
    """Return (train_indices, validation_indices, test_indices)."""
    dg = groups["drug"]
    tg = groups["target"]
    n = (fold + 1) % 6
    train, validation, test = [], [], []
    if setting == "cold_drug":
        for i in range(len(drugs)):
            g = dg[drugs[i]]
            if g == fold:
                test.append(i)
            elif g == n:
                validation.append(i)
            else:
                train.append(i)
    elif setting == "cold_target":
        for i in range(len(targets)):
            g = tg[targets[i]]
            if g == fold:
                test.append(i)
            elif g == n:
                validation.append(i)
            else:
                train.append(i)
    elif setting == "cold_both":
        for i in range(len(drugs)):
            gd = dg[drugs[i]]
            gt = tg[targets[i]]
            if gd == fold and gt == fold:
                test.append(i)
            elif gd == n and gt == n:
                validation.append(i)
            elif gd not in (fold, n) and gt not in (fold, n):
                train.append(i)
    else:
        raise ValueError(setting)
    return train, validation, test


def audit_cold(dataset_df, setting, fold, tr, te):
    """Entity-disjointness checks between the test and training sets."""
    tr_drugs = set(dataset_df["compound_iso_smiles"].iloc[tr])
    tr_targets = set(dataset_df["target_sequence"].iloc[tr])
    te_drugs = set(dataset_df["compound_iso_smiles"].iloc[te])
    te_targets = set(dataset_df["target_sequence"].iloc[te])
    if setting == "cold_drug":
        return {
            "test_drugs_unseen_in_train": len(te_drugs & tr_drugs) == 0,
            "test_drug_count": len(te_drugs),
        }
    if setting == "cold_target":
        return {
            "test_targets_unseen_in_train": len(te_targets & tr_targets) == 0,
            "test_target_count": len(te_targets),
        }
    return {
        "test_drugs_unseen_in_train": len(te_drugs & tr_drugs) == 0,
        "test_targets_unseen_in_train": len(te_targets & tr_targets) == 0,
        "test_drug_count": len(te_drugs),
        "test_target_count": len(te_targets),
    }


def prepare_dataset(dataset, seed):
    csv_path = MDS_ROOT / "data" / f"{dataset}_sixfold_all.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing canonical CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    drugs = df["compound_iso_smiles"].tolist()
    targets = df["target_sequence"].tolist()
    rng = random.Random(seed)
    # sorted() keeps the grouping identical across machines and Python
    # versions (set iteration order is hash-randomized and not stable).
    drug_groups = group_entities(rng, sorted(set(drugs)))
    target_groups = group_entities(rng, sorted(set(targets)))
    groups = {"drug": drug_groups, "target": target_groups}

    summaries = {}
    for setting in SETTINGS:
        setting_dir = OUTPUT_ROOT / "splits" / dataset / setting
        ensure_dir(setting_dir)
        fold_summaries = []
        for fold in range(6):
            tr, va, te = fold_partition(
                dataset, drugs, targets, groups, setting, fold)
            audit = audit_cold(df, setting, fold, tr, te)
            manifest = {
                "schema_version": 1,
                "dataset": dataset,
                "experiment": "cold_start",
                "setting": setting,
                "outer_fold": fold,
                "seed": seed,
                "protocol": (
                    "entity-level six-fold: test=entity group %d, "
                    "validation=group %d, train=other groups"
                    % (fold, (fold + 1) % 6)
                    if setting != "cold_both" else
                    "double-cold six-fold: test=(drug group %d, target group %d), "
                    "validation=(group %d, group %d), train=other groups"
                    % (fold, fold, (fold + 1) % 6, (fold + 1) % 6)),
                "drug_groups": drug_groups,
                "target_groups": target_groups,
                "train_indices": tr,
                "validation_indices": va,
                "test_indices": te,
                "sizes": {"train": len(tr), "validation": len(va),
                          "test": len(te)},
                "audit": audit,
            }
            out = setting_dir / f"fold_{fold}.json"
            with out.open("w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=1)
            fold_summaries.append({
                "fold": fold,
                "sizes": manifest["sizes"],
                "audit": audit,
            })
            print(f"{dataset} {setting} fold {fold}: "
                  f"train={len(tr)} val={len(va)} test={len(te)} "
                  f"audit={audit}", flush=True)
        summaries[setting] = fold_summaries

    summary_dir = OUTPUT_ROOT / dataset
    ensure_dir(summary_dir)
    with (summary_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "dataset": dataset,
            "experiment": "cold_start",
            "seed": seed,
            "n_samples": len(df),
            "n_unique_drugs": len(drug_groups),
            "n_unique_targets": len(target_groups),
            "settings": summaries,
        }, fh, indent=2, ensure_ascii=False)
    print(f"[OK] {dataset}: manifests written under {OUTPUT_ROOT / 'splits' / dataset}",
          flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build entity-level six-fold cold-start datasets.")
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

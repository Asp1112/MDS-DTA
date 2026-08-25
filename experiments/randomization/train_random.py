"""Randomization training entry (six-fold manifests, train-side shuffle).

The randomization control is applied only to the training split:
  * rand_x1 : protein (target) features are permuted among training samples;
  * rand_x2 : compound graphs (x / edge_index / c_size / edge_attr) are
              permuted among training samples;
  * rand_y  : affinity labels are permuted among training samples.
Validation and test sets are always kept intact, so the evaluation measures
how much genuine drug-target-label structure the model can learn from the
corrupted training set.

Example:
  python train_random.py --dataset davis --mode x2 --fold 0 \
      --model MDS_dta
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn


COMMON = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "common"
sys.path.insert(0, str(COMMON))

import exp_common  # noqa: E402
from exp_common import (  # noqa: E402
    add_experiment_args, build_model, check_done, load_manifest,
    load_mds_dataset, run_training, seed_everything,
)


EXPERIMENT = "randomization"
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
MODES = ["x1", "x2", "y"]


class PermutedDataset(torch.utils.data.Dataset):
    """Apply the stored train-side permutation to one input/label channel."""

    def __init__(self, base, train_indices, permutation, mode):
        self.base = base
        self.indices = list(train_indices)
        self.permutation = list(permutation)
        self.mode = mode

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data = self.base[self.indices[idx]]
        source = self.base[self.permutation[idx]]
        out = data.clone()
        if self.mode == "x1":
            out.target = source.target.clone()
        elif self.mode == "x2":
            out.x = source.x.clone()
            out.edge_index = source.edge_index.clone()
            if hasattr(data, "c_size"):
                out.c_size = source.c_size.clone()
            if hasattr(data, "edge_attr"):
                out.edge_attr = source.edge_attr.clone()
        elif self.mode == "y":
            out.y = source.y.clone()
        else:
            raise ValueError(self.mode)
        return out


def forward_loss(model, batch, loss_fn, device):
    batch = batch.to(device, non_blocking=True)
    output = model(batch)
    target = batch.y.view(-1, 1).float()
    return loss_fn(output, target), batch.y.view(-1), output


def predict(model, loader, device):
    model.eval()
    labels, preds = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            labels.append(batch.y.view(-1).cpu().numpy())
            preds.append(model(batch).view(-1).cpu().numpy())
    import numpy as np
    return np.concatenate(labels), np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser(
        description="Train the MDS model on six-fold randomization controls.")
    parser = add_experiment_args(parser, EXPERIMENT)
    parser.add_argument("--mode", choices=MODES, default="x2")
    parser.add_argument("--fold", type=int, required=True, choices=range(6))
    parser.add_argument("--model", default="MDS_dta",
                        help="CombinedDTA-family model module under models/.")
    parser.add_argument("--model-params", default=None)
    args = parser.parse_args()

    dataset = args.dataset.strip().lower()
    mode = args.mode
    fold = args.fold
    manifest_path = HERE / "data" / "splits" / dataset / ("rand_" + mode) / \
        f"fold_{fold}.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"Missing split manifest: {manifest_path} "
            "(run prepare_randomization.py first).")
    split = load_manifest(manifest_path)
    if not split.get("train_permutation"):
        raise SystemExit("Manifest has no train_permutation; regenerate it.")

    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    epochs = 2 if args.dry else args.epochs
    batch_size = args.batch_size or 256
    eval_batch_size = args.eval_batch_size or 256
    if args.dry:
        batch_size = min(64, batch_size)
        eval_batch_size = min(128, eval_batch_size)
    workers = 0 if args.dry else args.workers
    args.batch_size = batch_size
    args.eval_batch_size = eval_batch_size
    args.epochs = epochs

    base = load_mds_dataset(dataset)
    train_set = PermutedDataset(base, split["train_indices"],
                                split["train_permutation"], mode)
    from torch_geometric.loader import DataLoader as PyGDataLoader
    options = {"num_workers": workers, "pin_memory": device.type == "cuda",
               "persistent_workers": workers > 0}
    train_loader = PyGDataLoader(train_set, batch_size=batch_size, shuffle=True,
                                 **options)
    from exp_common import IndexedSubset
    validation_loader = PyGDataLoader(
        IndexedSubset(base, split["validation_indices"]),
        batch_size=eval_batch_size, shuffle=False, **options)
    test_loader = PyGDataLoader(
        IndexedSubset(base, split["test_indices"]),
        batch_size=eval_batch_size, shuffle=False, **options)

    model, model_module, model_class, applied, requested = build_model(
        args.model, args.model_params)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Randomization | dataset={dataset} mode=rand_{mode} fold={fold} "
          f"model={args.model} device={device} params={n_params:,}",
          flush=True)
    print(f"Split sizes train/val/test: {split['sizes']}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.scheduler_patience)
    loss_fn = nn.MSELoss()
    prefix = f"{model_class}_{dataset}_rand{mode}_fold{fold}"
    results_root = args.results_root
    if args.skip_done and results_root and check_done(results_root, prefix):
        print(f"[skip] {prefix}: completed run already exists.", flush=True)
        return

    run_training(
        args, EXPERIMENT, prefix, split, list(split["sizes"].values()),
        model, device, train_loader, validation_loader, test_loader,
        optimizer, scheduler, loss_fn, forward_loss, predict,
        model_module, model_class, applied, requested,
        {"family": "mds", "amp": True, "mode": "rand_" + mode, "fold": fold},
        manifest_path)


if __name__ == "__main__":
    main()


"""Few-shot (reduced-data) training entry (six-fold manifests).

The manifest already stores the subsampled training indices (50%, 25% or 10%
of the four training folds); the validation and test folds are kept complete.
This entry trains the MDS CombinedDTA-family model with the same
validation-driven recipe as the other experiments.

Example:
  python train_fewshot.py --dataset davis --setting fs25 --fold 0 \
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
    loaders_from_indices, run_training, seed_everything,
)


EXPERIMENT = "fewshot"
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = ["fs50", "fs25", "fs10"]


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
        description="Train the MDS model on six-fold few-shot subsets.")
    parser = add_experiment_args(parser, EXPERIMENT)
    parser.add_argument("--setting", choices=SETTINGS, default="fs50")
    parser.add_argument("--fold", type=int, required=True, choices=range(6))
    parser.add_argument("--model", default="MDS_dta",
                        help="CombinedDTA-family model module under models/.")
    parser.add_argument("--model-params", default=None)
    args = parser.parse_args()
    dataset = args.dataset.strip().lower()
    setting = args.setting
    fold = args.fold
    manifest_path = HERE / "data" / "splits" / dataset / setting / \
        f"fold_{fold}.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"Missing split manifest: {manifest_path} "
            "(run prepare_fewshot.py first).")
    split = load_manifest(manifest_path)
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
    train_loader, validation_loader, test_loader = loaders_from_indices(
        dataset, [split["train_indices"], split["validation_indices"],
                  split["test_indices"]], batch_size, eval_batch_size,
        workers, device)
    model, model_module, model_class, applied, requested = build_model(
        args.model, args.model_params)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Few-shot | dataset={dataset} setting={setting} fold={fold} "
          f"model={args.model} device={device} params={n_params:,}",
          flush=True)
    print(f"Split sizes train/val/test: {split['sizes']}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.scheduler_patience)
    loss_fn = nn.MSELoss()
    prefix = f"{model_class}_{dataset}_{setting}_fold{fold}"
    results_root = args.results_root
    if args.skip_done and results_root and check_done(results_root, prefix):
        print(f"[skip] {prefix}: completed run already exists.", flush=True)
        return
    run_training(
        args, EXPERIMENT, prefix, split, list(split["sizes"].values()),
        model, device, train_loader, validation_loader, test_loader,
        optimizer, scheduler, loss_fn, forward_loss, predict,
        model_module, model_class, applied, requested,
        {"family": "mds", "amp": True, "setting": setting, "fold": fold},
        manifest_path)


if __name__ == "__main__":
    main()

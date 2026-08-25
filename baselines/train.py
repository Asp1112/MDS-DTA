"""Train a CombinedDTA-family model on a fixed 4/1/1 six-part holdout split.

Dataset and model are selected on the command line, so you never have to edit
this file to switch experiments.

Examples:
  python train.py                                  # default: davis + combined_dta_lstmdrop
  python train.py --dataset kiba                   # KIBA with the default model
  python train.py --dataset bindingdb --model combined_dta
  python train.py --model combined_dta             # Davis with the base model
  python train.py --dataset kiba --model combined_dta_lstmdrop \
      --model-params '{"dropout": 0.2, "graph_steps": 2}'

Requirements per dataset:
  data/processed/<dataset>_sixfold_all.pt          (see prepare_sixfold_data.py)
  splits/<dataset>/fixed_six_part_split.json       (same six-part protocol as Davis)

The default training configuration is the exact formal Davis configuration:
seed 42, batch 256, lr 1e-4, weight decay 0, gradient clip 5, scheduler
patience 60, early stopping 150, top-3 checkpoint averaging.
"""

import argparse
import csv
import importlib
import inspect
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from utils import TestbedDataset, ci, mae, mse, pearson, r2, rm2, rmse, spearman


# Reproducible formal-training configuration (identical to the Davis run)
SEED = 42
DEVICE = "cuda:0"
EPOCHS = 1000
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 5.0
SCHEDULER_PATIENCE = 60
EARLY_STOPPING_PATIENCE = 150
TOP_K = 3
NUM_WORKERS = 4

RESULTS_DIR = Path("results")

# Default model hyper-parameters; only the keys accepted by the chosen model
# constructor are actually passed, so any model with a compatible signature
# works without code changes.
DEFAULT_MODEL_PARAMS = {
    "dropout": 0.1,
    "graph_hidden": 64,
    "graph_steps": 3,
    "lstm_layers": 2,
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(y_true, y_pred):
    return {
        "mse": float(mse(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "pearson": float(pearson(y_true, y_pred)),
        "spearman": float(spearman(y_true, y_pred)),
        "ci": float(ci(y_true, y_pred)),
        "r2": float(r2(y_true, y_pred)),
        "rm2": float(rm2(y_true, y_pred)),
        "mae": float(mae(y_true, y_pred)),
    }


def resolve_model(model_spec):
    """Import a model module and return (module, model_class).

    model_spec is a module name under models/ (e.g. combined_dta,
    combined_dta_lstmdrop) or any dotted path importable from the project root.
    The class is picked automatically: first the class whose name matches the
    module name, then the first CombinedDTA* nn.Module, then any nn.Module.
    """
    try:
        module = importlib.import_module("models." + model_spec)
    except ModuleNotFoundError:
        try:
            module = importlib.import_module(model_spec)
        except ModuleNotFoundError as exc:
            raise SystemExit(
                f"Model module not found: {model_spec!r} "
                f"(tried models.{model_spec} and {model_spec}). "
                f"Put the model file under models/ and pass its name.") from exc

    candidates = [
        value for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, nn.Module)
        and getattr(value, "__module__", None) == module.__name__
    ]
    normalized_module = module.__name__.rsplit(".", 1)[-1].replace("_", "").lower()
    for candidate in candidates:
        if candidate.__name__.replace("_", "").lower() == normalized_module:
            return module, candidate
    for candidate in candidates:
        if candidate.__name__.startswith("CombinedDTA"):
            return module, candidate
    if candidates:
        return module, candidates[0]
    raise SystemExit(f"No nn.Module class found in model module {model_spec!r}.")


def build_model(model_spec, model_params_json):
    module, model_class = resolve_model(model_spec)
    params = dict(DEFAULT_MODEL_PARAMS)
    if model_params_json:
        try:
            extra = json.loads(model_params_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--model-params is not valid JSON: {exc}") from exc
        if not isinstance(extra, dict):
            raise SystemExit("--model-params must be a JSON object, e.g. '{\"dropout\": 0.2}'")
        params.update(extra)

    signature = inspect.signature(model_class.__init__)
    accepted = {
        name: value for name, value in params.items() if name in signature.parameters
    }
    model = model_class(**accepted)
    return model, module.__name__, model_class.__name__, accepted, params


def make_loaders(device, dataset_name, split_file):
    data_file = dataset_name + "_sixfold_all"
    if not split_file.exists():
        raise SystemExit(
            f"Split file not found: {split_file}. Run "
            f"prepare_sixfold_data.py first (or check the dataset name).")
    dataset = TestbedDataset(root="data", dataset=data_file)
    with split_file.open(encoding="utf-8") as file:
        split = json.load(file)

    groups = [split["train_indices"], split["validation_indices"], split["test_indices"]]
    indices = sum(groups, [])
    if len(indices) != len(dataset) or len(set(indices)) != len(dataset):
        raise ValueError("Dataset and fixed split are inconsistent.")

    options = {
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
    }
    train = DataLoader(
        Subset(dataset, groups[0]), batch_size=BATCH_SIZE, shuffle=True, **options)
    validation = DataLoader(
        Subset(dataset, groups[1]), batch_size=EVAL_BATCH_SIZE, shuffle=False, **options)
    test = DataLoader(
        Subset(dataset, groups[2]), batch_size=EVAL_BATCH_SIZE, shuffle=False, **options)
    return train, validation, test, [len(group) for group in groups], split_file


def train_epoch(model, loader, optimizer, loss_function, scaler, device, epoch):
    model.train()
    loss_sum = torch.zeros((), device=device)
    max_grad = torch.zeros((), device=device)
    nonfinite_batches = torch.zeros((), dtype=torch.int64, device=device)

    progress = tqdm(loader, desc=f"Train {epoch:04d}")
    for batch_index, batch in enumerate(progress):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            loss = loss_function(model(batch), batch.y.view(-1, 1).float())

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        finite = torch.isfinite(grad)
        max_grad = torch.maximum(
            max_grad, torch.where(finite, grad.detach(), torch.zeros_like(grad)))
        nonfinite_batches += (~finite).to(torch.int64)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.detach()
        if batch_index % 100 == 0:
            progress.set_postfix(loss=f"{loss.item():.3f}", refresh=False)

    return (
        float((loss_sum / len(loader)).item()),
        float(max_grad.item()),
        int(nonfinite_batches.item()),
    )


def predict(model, loader, device):
    model.eval()
    labels, predictions = [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Validate", leave=False):
            batch = batch.to(device, non_blocking=True)
            labels.append(batch.y.view(-1))
            predictions.append(model(batch).view(-1))
    return (
        torch.cat(labels).cpu().numpy(),
        torch.cat(predictions).cpu().numpy(),
    )


def retain_top_k(top_k, model, run_dir, epoch, validation_mse):
    if len(top_k) == TOP_K and validation_mse >= top_k[-1]["mse"]:
        return

    path = run_dir / f"checkpoint_epoch{epoch:04d}_mse{validation_mse:.6f}.pt"
    torch.save({"epoch": epoch, "mse": validation_mse,
                "model_state_dict": model.state_dict()}, path)
    top_k.append({"epoch": epoch, "mse": validation_mse, "path": path})
    top_k.sort(key=lambda item: item["mse"])
    if len(top_k) > TOP_K:
        top_k.pop()["path"].unlink()


def averaged_state(top_k):
    states = [
        torch.load(item["path"], map_location="cpu", weights_only=True)["model_state_dict"]
        for item in top_k
    ]
    average = {}
    for key in states[0]:
        tensors = [state[key] for state in states]
        average[key] = (
            torch.stack(tensors).mean(0)
            if tensors[0].is_floating_point() else tensors[0]
        )
    return average


def test_model(model, state, loader, device):
    model.load_state_dict(state)
    labels, predictions = predict(model, loader, device)
    return metrics(labels, predictions), labels, predictions


def main():
    parser = argparse.ArgumentParser(
        description="Train a CombinedDTA-family model on a fixed six-part holdout split.")
    parser.add_argument(
        "--dataset", default="davis",
        help="Dataset name (case-insensitive). Expects "
             "data/processed/<dataset>_sixfold_all.pt and "
             "splits/<dataset>/fixed_six_part_split.json.")
    parser.add_argument(
        "--model", default="combined_dta_lstmdrop",
        help="Model module under models/ (e.g. combined_dta, "
             "combined_dta_lstmdrop) or any dotted import path. "
             "The nn.Module class is resolved automatically.")
    parser.add_argument(
        "--model-params", default=None,
        help="Optional JSON object overriding model hyper-parameters, "
             "e.g. '{\"dropout\": 0.2, \"graph_steps\": 2}'.")
    parser.add_argument(
        "--test-fold", type=int, default=None,
        help="Run fold <N> of the six-fold rotation (0..5): fold N is the "
             "one-time test set, fold (N+1) mod 6 is the validation set, and "
             "the other four folds train (splits/<dataset>/fold_<N>.json). "
             "Omit to use the fixed 4/1/1 split.")
    parser.add_argument(
        "--results-root", default=None,
        help="Output directory for the run (default: results/). Rotation runs "
             "should use a dedicated root such as results/cv/<dataset> so they "
             "are not confused with fixed-split runs.")
    args = parser.parse_args()

    dataset_name = args.dataset.strip().lower()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    seed_everything(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model, model_module_name, model_class_name, applied_params, requested_params = \
        build_model(args.model, args.model_params)
    model = model.to(device)
    print(f"Dataset: {dataset_name} | Model: {model_module_name} "
          f"({model_class_name}) | Device: {device}")
    print(f"Applied model params: {applied_params}")

    if args.test_fold is not None:
        if not 0 <= args.test_fold <= 5:
            raise SystemExit("--test-fold must be between 0 and 5.")
        split_file = Path("splits") / dataset_name / f"fold_{args.test_fold}.json"
    else:
        split_file = Path("splits") / dataset_name / "fixed_six_part_split.json"
    train_loader, validation_loader, test_loader, sizes, split_file = make_loaders(
        device, dataset_name, split_file)
    print(f"Split train/validation/test: {sizes}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=4096.0)
    loss_function = nn.MSELoss()

    results_root = Path(args.results_root) if args.results_root else RESULTS_DIR
    tag = f"_fold{args.test_fold}" if args.test_fold is not None else ""
    run_dir = results_root / time.strftime(
        f"{model_class_name}_{dataset_name}{tag}_%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    saved_config = {
        "dataset": dataset_name,
        "model_module": model_module_name,
        "model_class": model_class_name,
        "model_parameters": applied_params,
        "requested_model_parameters": requested_params,
        "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRAD_CLIP_NORM,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "top_k": TOP_K,
        "eval_batch_size": EVAL_BATCH_SIZE, "split_sizes": sizes,
        "test_fold": args.test_fold,
        "results_root": str(results_root),
        "split_protocol": (
            f"six-fold rotation: test=fold {args.test_fold}, "
            f"validation=fold {(args.test_fold + 1) % 6}, train=other four folds"
            if args.test_fold is not None else
            "one fixed 4/1/1 split of six parts; no fold rotation"),
    }
    (run_dir / "config.json").write_text(
        json.dumps(saved_config, indent=2), encoding="utf-8")
    shutil.copyfile(split_file, run_dir / "split_indices.json")

    fields = ["epoch", "train_loss", "mse", "rmse", "pearson", "spearman",
              "ci", "r2", "rm2", "mae", "lr", "max_grad", "amp_overflows", "seconds"]
    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=fields).writeheader()

    top_k, best_mse, stale_epochs = [], float("inf"), 0
    epoch_times, total_overflows = [], 0
    best_validation_metrics = None

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss, max_grad, overflows = train_epoch(
            model, train_loader, optimizer, loss_function, scaler, device, epoch)
        y_true, y_pred = predict(model, validation_loader, device)
        validation_mse = float(np.mean((y_true - y_pred) ** 2))
        scheduler.step(validation_mse)
        retain_top_k(top_k, model, run_dir, epoch, validation_mse)

        improved = validation_mse < best_mse
        if improved:
            best_mse = validation_mse
            best_validation_metrics = metrics(y_true, y_pred)
            stale_epochs = 0
        else:
            stale_epochs += 1

        seconds = time.time() - start
        epoch_times.append(seconds)
        total_overflows += overflows
        detailed = best_validation_metrics if improved else {}
        row = {
            "epoch": epoch, "train_loss": train_loss, "mse": validation_mse,
            **{name: detailed.get(name, "") for name in
               ("rmse", "pearson", "spearman", "ci", "r2", "rm2", "mae")},
            "lr": optimizer.param_groups[0]["lr"], "max_grad": max_grad,
            "amp_overflows": overflows, "seconds": round(seconds, 2),
        }
        with history_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=fields).writerow(row)
        print(
            f"Epoch {epoch}: val MSE={validation_mse:.6f}, best={best_mse:.6f}, "
            f"lr={row['lr']:.2e}, grad={max_grad:.2f}, overflow={overflows}, {seconds:.1f}s")

        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    summary = {
        "dataset": dataset_name,
        "model_class": model_class_name,
        "epochs_completed": len(epoch_times),
        "best_epoch": top_k[0]["epoch"],
        "best_validation_metrics": best_validation_metrics,
        "top3_validation_mse": [
            {"epoch": item["epoch"], "mse": item["mse"]} for item in top_k],
        "mean_epoch_seconds": float(np.mean(epoch_times)),
        "total_amp_overflows": total_overflows,
    }
    (run_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    best_checkpoint = torch.load(top_k[0]["path"], map_location="cpu", weights_only=True)
    best_result, labels, best_predictions = test_model(
        model, best_checkpoint["model_state_dict"], test_loader, device)
    top3_result, _, top3_predictions = test_model(
        model, averaged_state(top_k), test_loader, device)
    best_result["selected_epoch"] = top_k[0]["epoch"]
    top3_result["averaged_epochs"] = [item["epoch"] for item in top_k]
    test_results = {"best_checkpoint": best_result, "top3_average": top3_result}
    (run_dir / "test_metrics.json").write_text(
        json.dumps(test_results, indent=2), encoding="utf-8")
    np.savez_compressed(
        run_dir / "test_predictions.npz", labels=labels,
        best_checkpoint=best_predictions, top3_average=top3_predictions)
    print(json.dumps(test_results, indent=2))
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()

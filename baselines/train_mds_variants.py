"""Train the three CombinedDTA fusion variants on the formal Davis holdout.

The training protocol follows the formal Davis configuration used in the
project: seed 42, lr 1e-4, weight decay 0, grad clip 5, ReduceLROnPlateau
patience 60, early stopping 150, top-3 checkpoint averaging, and the fixed
4/1/1 six-part split stored under splits/davis/fixed_six_part_split.json.
The batch size is 128 (the reference used 256) because the token-level
cross-attention variants do not fit on the 8 GB RTX 4060 Ti at 256.

Usage:
  python train_mds_variants.py --model v2
  python train_mds_variants.py --model b
  python train_mds_variants.py --model c --model-params '{"dropout": 0.2}'
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

# Reduce caching-allocator fragmentation on the 8 GB GPU before torch starts.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils import TestbedDataset, ci, mae, mse, pearson, r2, rm2, rmse, spearman


# Formal Davis configuration (mirrors the reference runs; batch reduced to 128)
SEED = 42
DEVICE = "cuda:0"
EPOCHS = 1000
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 5.0
SCHEDULER_PATIENCE = 60
EARLY_STOPPING_PATIENCE = 150
TOP_K = 3
NUM_WORKERS = 4

DATA_ROOT = Path(os.environ.get("MDS_DATA_ROOT", ROOT / "data"))
SPLIT_FILE = ROOT / "splits" / "davis" / "fixed_six_part_split.json"
RESULTS_DIR = ROOT / "results"
DATASET_FILE = "davis_sixfold_all"

MODEL_MAP = {
    "v2": ("models.combined_dta_v2", "CombinedDTAV2"),
    "b": ("models.combined_dta_v2B", "CombinedDTAV2B"),
    "c": ("models.combined_dta_v2C", "CombinedDTAV2C"),
}

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


def build_model(model_key, model_params_json):
    if model_key not in MODEL_MAP:
        raise SystemExit(
            f"Unknown --model {model_key!r}; choose one of {sorted(MODEL_MAP)}")
    module_name, class_name = MODEL_MAP[model_key]
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    params = dict(DEFAULT_MODEL_PARAMS)
    if model_params_json:
        try:
            extra = json.loads(model_params_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--model-params is not valid JSON: {exc}") from exc
        if not isinstance(extra, dict):
            raise SystemExit("--model-params must be a JSON object")
        params.update(extra)
    signature = inspect.signature(model_class.__init__)
    accepted = {
        name: value for name, value in params.items()
        if name in signature.parameters
    }
    model = model_class(**accepted)
    return model, module_name, class_name, accepted, params


def make_loaders(device):
    dataset = TestbedDataset(root=str(DATA_ROOT), dataset=DATASET_FILE)
    with SPLIT_FILE.open(encoding="utf-8") as file:
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
    return train, validation, test, [len(group) for group in groups]


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
    parser = argparse.ArgumentParser(description="Train CombinedDTA fusion variants on Davis.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_MAP),
                        help="v2 = improved token-level cross-attention; "
                             "b = gated MLP mixing (no attention); "
                             "c = token-level attention + mean/max pooling")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Maximum number of epochs (default 1000).")
    parser.add_argument("--model-params", default=None,
                        help="Optional JSON object overriding model hyper-parameters.")
    args = parser.parse_args()
    epochs = args.epochs

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    seed_everything(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model, module_name, class_name, applied, requested = build_model(
        args.model, args.model_params)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Dataset: davis | Model: {module_name} ({class_name}) | Device: {device}")
    print(f"Applied model params: {applied}")
    print(f"Trainable parameters: {n_params:,} ({n_params / 1e6:.2f} M)")

    train_loader, validation_loader, test_loader, sizes = make_loaders(device)
    print(f"Fixed split train/validation/test: {sizes}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=4096.0)
    loss_function = nn.MSELoss()

    run_dir = RESULTS_DIR / time.strftime(f"{class_name}_davis_%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    saved_config = {
        "dataset": "davis",
        "model_module": module_name,
        "model_class": class_name,
        "model_parameters": applied,
        "requested_model_parameters": requested,
        "seed": SEED, "epochs": epochs, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRAD_CLIP_NORM,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "top_k": TOP_K,
        "eval_batch_size": EVAL_BATCH_SIZE, "split_sizes": sizes,
        "split_protocol": "one fixed 4/1/1 split of six parts; no fold rotation",
        "data_file": str(DATA_ROOT / "processed" / (DATASET_FILE + ".pt")),
        "split_file": str(SPLIT_FILE),
    }
    (run_dir / "config.json").write_text(json.dumps(saved_config, indent=2), encoding="utf-8")
    shutil.copyfile(SPLIT_FILE, run_dir / "split_indices.json")

    fields = ["epoch", "train_loss", "mse", "rmse", "pearson", "spearman",
              "ci", "r2", "rm2", "mae", "lr", "max_grad", "amp_overflows", "seconds"]
    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=fields).writeheader()

    top_k, best_mse, stale_epochs = [], float("inf"), 0
    epoch_times, total_overflows = [], 0
    best_validation_metrics = None

    for epoch in range(1, epochs + 1):
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
            f"lr={row['lr']:.2e}, grad={max_grad:.2f}, overflow={overflows}, "
            f"{seconds:.1f}s", flush=True)

        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break

    summary = {
        "dataset": "davis",
        "model_class": class_name,
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
    print(json.dumps(test_results, indent=2), flush=True)
    print(f"Results saved to {run_dir}", flush=True)


if __name__ == "__main__":
    main()

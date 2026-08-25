"""Shared helpers for the six-fold experiment families (cold start,
randomization, few-shot) deployed on the MDS servers.

All three experiment folders use this module for:
  * deterministic seeds and the same metric set as the six-fold CV runs;
  * split-manifest loading (train / validation / test index lists);
  * the CombinedDTA-family model resolution (identical to train_test.py);
  * the validation-driven training loop (plateau LR, AMP, top-3 averaging,
    early stopping, one-time test evaluation) and the standard result layout
    (config.json / split_indices.json / validation_summary.json /
    test_metrics.json / test_predictions.npz / history.csv).

The dataset root defaults to /root/mds and can be overridden with the
MDS_ROOT environment variable (used for local verification runs).
"""

import argparse
import csv
import importlib
import inspect
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader


MDS_ROOT = Path(os.environ.get("MDS_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
EXPERIMENTS_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(MDS_ROOT) not in sys.path:
    sys.path.insert(0, str(MDS_ROOT))

TOP_K = 3
SEED = 42
EPOCHS = 1000
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 5.0
SCHEDULER_PATIENCE = 60
EARLY_STOPPING_PATIENCE = 100
NUM_WORKERS = 4

DEFAULT_MODEL_PARAMS = {
    "dropout": 0.1,
    "graph_hidden": 64,
    "graph_steps": 3,
    "lstm_layers": 2,
}


class SavedGraphDataset(InMemoryDataset):
    """Load a collated (data, slices) PyG file and index it like a dataset."""
    def __init__(self, path):
        super().__init__()
        self.data, self.slices = torch.load(
            path, map_location="cpu", weights_only=False)
    def len(self):
        return int(self.slices["y"][-1].item())


class IndexedSubset(torch.utils.data.Dataset):
    """Subset that attaches the global dataset index to each sample.
    Needed by models that look up extra per-sample information (for example
    precomputed ESM features) aligned with the full dataset order.
    """
    def __init__(self, base, indices):
        self.base = base
        self.indices = list(indices)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        data = self.base[self.indices[idx]]
        data.esm_idx = torch.tensor([self.indices[idx]])
        return data


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ci_index(y, f):
    """Concordance index with O(n log n) tie handling (same as baselines)."""
    ind = np.argsort(y, kind="mergesort")
    y = y[ind]
    f = f[ind]
    uniq_y, counts = np.unique(y, return_counts=True)
    f_unique = np.unique(f)
    ranks = np.searchsorted(f_unique, f, side="left")
    m = f_unique.size
    bit = np.zeros(m + 1, dtype=np.int64)
    freq = np.zeros(m, dtype=np.int64)
    def update(idx, delta=1):
        i = idx + 1
        while i <= m:
            bit[i] += delta
            i += i & -i
    def query(idx):
        if idx < 0:
            return 0
        s = 0
        i = idx + 1
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s
    start = total_processed = comparable = concordant = ties = 0
    for c in counts:
        end = start + c
        group_ranks = ranks[start:end]
        prev = total_processed
        if prev > 0:
            comparable += prev * c
        for r in group_ranks:
            concordant += query(r - 1)
            ties += freq[r]
        for r in group_ranks:
            update(r)
            freq[r] += 1
        total_processed += c
        start = end
    if comparable == 0:
        return float("nan")
    return float(concordant + 0.5 * ties) / comparable


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(math.sqrt(mse))
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    spearman = float(stats.spearmanr(y_true, y_pred)[0])
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    slope = float(np.sum(y_true * y_pred) / max(np.sum(y_pred ** 2), 1e-12))
    y_pred_origin = slope * y_pred
    r2_origin = 1.0 - float(np.sum((y_true - y_pred_origin) ** 2)) / ss_tot \
        if ss_tot > 0 else float("nan")
    rm2 = r2 * (1.0 - math.sqrt(abs(r2 - r2_origin))) if r2 <= 1.0 else r2
    return {
        "mse": mse,
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
        "ci": ci_index(y_true, y_pred),
        "r2": r2,
        "rm2": rm2,
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


def load_manifest(path):
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def make_run_dir(results_root, prefix):
    root = Path(results_root) if results_root else Path("results")
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / time.strftime(f"{prefix}_%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    return run_dir


def retain_top_k(top_k, model, run_dir, epoch, validation_mse, top_k_size=TOP_K):
    if len(top_k) == top_k_size and validation_mse >= top_k[-1]["mse"]:
        return
    path = run_dir / f"checkpoint_epoch{epoch:04d}_mse{validation_mse:.6f}.pt"
    torch.save({"epoch": epoch, "mse": validation_mse,
                "model_state_dict": model.state_dict()}, path)
    top_k.append({"epoch": epoch, "mse": validation_mse, "path": path})
    top_k.sort(key=lambda item: item["mse"])
    if len(top_k) > top_k_size:
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


def write_history(run_dir, rows, fields):
    path = run_dir / "history.csv"
    path.write_text("", encoding="utf-8")
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_model(model_spec):
    """Import a CombinedDTA-family model module and return (module, class)."""
    try:
        module = importlib.import_module("models." + model_spec)
    except ModuleNotFoundError:
        try:
            module = importlib.import_module("models.ablation." + model_spec)
        except ModuleNotFoundError:
            module = importlib.import_module(model_spec)
    except ModuleNotFoundError:
        try:
            module = importlib.import_module(model_spec)
        except ModuleNotFoundError as exc:
            raise SystemExit(
                f"Model module not found: {model_spec!r} "
                f"(tried models.{model_spec} and {model_spec}).") from exc
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
            raise SystemExit("--model-params must be a JSON object")
        params.update(extra)
    signature = inspect.signature(model_class.__init__)
    accepted = {
        name: value for name, value in params.items()
        if name in signature.parameters
    }
    model = model_class(**accepted)
    return model, module.__name__, model_class.__name__, accepted, params


def load_mds_dataset(dataset):
    """TestbedDataset over the canonical <dataset>_sixfold_all.pt graph file."""
    from utils import TestbedDataset
    return TestbedDataset(root=str(MDS_ROOT / "data"), dataset=dataset + "_sixfold_all")


def load_rows(dataset):
    csv_path = MDS_ROOT / "data" / f"{dataset}_sixfold_all.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing canonical CSV: {csv_path}")
    return pd.read_csv(csv_path)


def default_results_root(experiment):
    return MDS_ROOT / "results" / "experiments" / experiment


def add_experiment_args(parser, experiment):
    parser.add_argument("--dataset", default="davis",
                        help="davis / kiba / bindingdb")
    parser.add_argument("--results-root", default=None,
                        help=f"Output root (default: "
                             f"{default_results_root(experiment)}).")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="Default resolved per model family.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--scheduler-patience", type=int, default=SCHEDULER_PATIENCE)
    parser.add_argument("--early-stopping-patience", type=int,
                        default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--dry", action="store_true",
                        help="Two-epoch smoke run (small batches).")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip when a completed test_metrics.json already "
                             "exists for this experiment/fold/model.")
    return parser


def loaders_from_indices(dataset, indices, batch_size, eval_batch_size,
                         workers, device):
    """Build train/validation/test DataLoaders over the MDS graph file."""
    base = load_mds_dataset(dataset)
    options = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    train = DataLoader(IndexedSubset(base, indices[0]), batch_size=batch_size,
                       shuffle=True, **options)
    validation = DataLoader(IndexedSubset(base, indices[1]),
                            batch_size=eval_batch_size, shuffle=False, **options)
    test = DataLoader(IndexedSubset(base, indices[2]),
                      batch_size=eval_batch_size, shuffle=False, **options)
    return train, validation, test


def check_done(results_root, prefix):
    if not results_root:
        return False
    pattern = Path(results_root) / (prefix + "_*") / "test_metrics.json"
    return bool(list(Path(results_root).glob(prefix + "_*/test_metrics.json")))


def run_training(args, experiment, prefix, split, split_sizes, model,
                 device, train_loader, validation_loader, test_loader,
                 optimizer, scheduler, loss_fn, forward_loss, predict_fn,
                 model_module_name, model_class_name, applied_params,
                 requested_params, family_info, split_source):
    """Validation-driven training + one-time test evaluation.
    forward_loss(model, batch, loss_fn, device) -> (loss, target, output)
    predict_fn(model, batch, device) -> (labels, preds)
    """
    results_root = Path(args.results_root) if args.results_root else \
        default_results_root(experiment)
    run_dir = make_run_dir(results_root, prefix)
    config = {
        "dataset": args.dataset,
        "experiment": experiment,
        "model_module": model_module_name,
        "model_class": model_class_name,
        "model_parameters": applied_params,
        "requested_model_parameters": requested_params,
        "loss": "MSE",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": "plateau",
        "scheduler_patience": args.scheduler_patience,
        "early_stopping_patience": args.early_stopping_patience,
        "top_k": TOP_K,
        "split_sizes": split_sizes,
        "results_root": str(results_root),
        "split_source": str(split_source),
        "family": family_info,
        **{k: v for k, v in split.items()
           if k not in ("train_indices", "validation_indices", "test_indices",
                        "train_permutation", "drug_groups", "target_groups")},
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")
    if isinstance(split_source, (str, Path)):
        shutil.copyfile(split_source, run_dir / "split_indices.json")
    else:
        (run_dir / "split_indices.json").write_text(
            json.dumps(split, indent=2), encoding="utf-8")
    fields = ["epoch", "train_loss", "mse", "rmse", "pearson", "spearman",
              "ci", "r2", "rm2", "mae", "lr", "max_grad", "seconds"]
    history = []
    top_k = []
    best_mse = float("inf")
    best_validation_metrics = None
    stale = 0
    early_stop = False
    use_amp = bool(family_info.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=4096.0)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        loss_sum = 0.0
        max_grad = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss, _, _ = forward_loss(model, batch, loss_fn, device)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRAD_CLIP_NORM)
                max_grad = max(max_grad, float(grad))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, _, _ = forward_loss(model, batch, loss_fn, device)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRAD_CLIP_NORM)
                max_grad = max(max_grad, float(grad))
                optimizer.step()
            loss_sum += float(loss.detach())
        train_loss = loss_sum / max(1, len(train_loader))
        y_true, y_pred = predict_fn(model, validation_loader, device)
        validation_mse = float(np.mean((y_true - y_pred) ** 2))
        scheduler.step(validation_mse)
        retain_top_k(top_k, model, run_dir, epoch, validation_mse)
        improved = validation_mse < best_mse
        if improved:
            best_mse = validation_mse
            best_validation_metrics = metrics(y_true, y_pred)
            stale = 0
        else:
            stale += 1
        seconds = time.time() - start
        if hasattr(optimizer, "optimizer"):
            lr = optimizer.optimizer.param_groups[0]["lr"]
        else:
            lr = optimizer.param_groups[0]["lr"]
        detailed = best_validation_metrics if improved else {}
        row = {
            "epoch": epoch, "train_loss": train_loss, "mse": validation_mse,
            **{name: detailed.get(name, "") for name in
               ("rmse", "pearson", "spearman", "ci", "r2", "rm2", "mae")},
            "lr": lr, "max_grad": max_grad, "seconds": round(seconds, 2),
        }
        history.append(row)
        print(f"Epoch {epoch}: train_loss={train_loss:.6f} "
              f"val_mse={validation_mse:.6f} best={best_mse:.6f} "
              f"({seconds:.1f}s)", flush=True)
        if stale >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}.", flush=True)
            early_stop = True
            break
    write_history(run_dir, history, fields)
    summary = {
        "dataset": args.dataset,
        "experiment": experiment,
        "model_class": model_class_name,
        "epochs_completed": len(history),
        "best_epoch": top_k[0]["epoch"],
        "best_validation_metrics": best_validation_metrics,
        "top3_validation_mse": [
            {"epoch": item["epoch"], "mse": item["mse"]} for item in top_k],
        "mean_epoch_seconds": float(np.mean([h.get("seconds", 0.0) for h in history])),
        "early_stopped": early_stop,
    }
    (run_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    def evaluate(state):
        model.load_state_dict(state)
        labels, preds = predict_fn(model, test_loader, device)
        return metrics(labels, preds), labels, preds
    best_state = torch.load(top_k[0]["path"], map_location="cpu",
                            weights_only=True)["model_state_dict"]
    best_result, labels, best_pred = evaluate(best_state)
    best_result["selected_epoch"] = top_k[0]["epoch"]
    top3_state = averaged_state(top_k)
    top3_result, _, top3_pred = evaluate(top3_state)
    top3_result["averaged_epochs"] = [item["epoch"] for item in top_k]
    test_results = {"best_checkpoint": best_result, "top3_average": top3_result}
    (run_dir / "test_metrics.json").write_text(
        json.dumps(test_results, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "test_predictions.npz",
                        labels=labels, best_checkpoint=best_pred,
                        top3_average=top3_pred)
    print(json.dumps(test_results, indent=2), flush=True)
    print(f"Results saved to {run_dir}", flush=True)
    return test_results

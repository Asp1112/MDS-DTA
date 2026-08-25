import argparse
import csv
import importlib
import inspect
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils import TestbedDataset, ci, mae, mse, pearson, r2, rm2, rmse, spearman


SEED = 42
DEVICE = "cuda:0"
EPOCHS = 1000
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 5.0
SCHEDULER = "plateau"
WARMUP_EPOCHS = 5
COSINE_TMAX = 300
ETA_MIN = 1e-5
SCHEDULER_PATIENCE = 60
EARLY_STOPPING_PATIENCE = 100
TOP_K = 3
NUM_WORKERS = 4

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
    """Import a model module and return (module, model_class)."""
    try:
        module = importlib.import_module("models." + model_spec)
    except ModuleNotFoundError:
        try:
            module = importlib.import_module("models.ablation." + model_spec)
        except ModuleNotFoundError:
            try:
                module = importlib.import_module(model_spec)
            except ModuleNotFoundError as exc:
                raise SystemExit(
                    f"Model module not found: {model_spec!r} "
                    f"(tried models.{model_spec}, models.ablation.{model_spec} "
                    f"and {model_spec}).") from exc
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
        if candidate.__name__.startswith("MDSDTA"):
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


def build_scheduler(optimizer, mode, epochs, warmup_epochs, patience, factor,
                    cosine_tmax=COSINE_TMAX, eta_min=ETA_MIN):
    if mode == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience)
        return scheduler, "plateau"
    warmup_epochs = max(1, int(warmup_epochs))
    cosine_tmax = max(warmup_epochs + 1, int(cosine_tmax))
    base_lr = optimizer.param_groups[0]["lr"]
    eta_ratio = max(1e-6, float(eta_min) / base_lr)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        t = min(1.0, (epoch - warmup_epochs) / max(1, cosine_tmax - warmup_epochs))
        return max(eta_ratio, 0.5 * (1.0 + math.cos(math.pi * t)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler, "lambda"


def make_loaders(device, dataset_name, split_file, batch_size=BATCH_SIZE,
                 eval_batch_size=EVAL_BATCH_SIZE):
    data_file = dataset_name + "_sixfold_all"
    if not split_file.exists():
        raise SystemExit(
            f"Split file not found: {split_file}. Run create_data.py first.")
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
        Subset(dataset, groups[0]), batch_size=batch_size, shuffle=True, **options)
    validation = DataLoader(
        Subset(dataset, groups[1]), batch_size=eval_batch_size, shuffle=False, **options)
    test = DataLoader(
        Subset(dataset, groups[2]), batch_size=eval_batch_size, shuffle=False, **options)
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
        float((loss_sum / max(1, len(loader))).item()),
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
        description="Train MDSDTA-family models (plain MSE, clean recipe).")
    parser.add_argument("--dataset", default="davis")
    parser.add_argument("--model", default="MDS_dta")
    parser.add_argument("--model-params", default=None,
                        help="Optional JSON object overriding model hyper-parameters.")
    parser.add_argument("--test-fold", type=int, default=None,
                        help="Run fold <N> of the six-fold rotation (0..5): fold N "
                             "is the one-time test set, fold (N+1) mod 6 is the "
                             "validation set, and the other four folds train "
                             "(splits/<dataset>/fold_<N>.json). Omit to use the "
                             "fixed 4/1/1 split.")
    parser.add_argument("--results-root", default=None,
                        help="Output directory for the run (default: results/). "
                             "Rotation runs should use a dedicated root such as "
                             "results/cv/<dataset>.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip-norm", type=float, default=GRAD_CLIP_NORM)
    parser.add_argument("--scheduler", choices=["plateau", "cosine"],
                        default=SCHEDULER)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--cosine-tmax", type=int, default=COSINE_TMAX)
    parser.add_argument("--eta-min", type=float, default=ETA_MIN)
    parser.add_argument("--scheduler-patience", type=int, default=SCHEDULER_PATIENCE)
    parser.add_argument("--early-stopping-patience", type=int,
                        default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    dataset_name = args.dataset.strip().lower()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    seed_everything(args.seed)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model, model_module_name, model_class_name, applied_params, requested_params = \
        build_model(args.model, args.model_params)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Dataset: {dataset_name} | Model: {model_module_name} "
          f"({model_class_name}) | Device: {device}")
    print(f"Applied model params: {applied_params}")
    print(f"Trainable parameters: {n_params:,} ({n_params / 1e6:.2f} M)")
    if args.test_fold is not None:
        if not 0 <= args.test_fold <= 5:
            raise SystemExit("--test-fold must be between 0 and 5.")
        split_file = Path("data/splits") / dataset_name / f"fold_{args.test_fold}.json"
    else:
        split_file = Path("data/splits") / dataset_name / "fixed_six_part_split.json"
    train_loader, validation_loader, test_loader, sizes, split_file = make_loaders(
        device, dataset_name, split_file, args.batch_size, args.eval_batch_size)
    print(f"Split train/validation/test: {sizes}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler, scheduler_kind = build_scheduler(
        optimizer, args.scheduler, args.epochs, args.warmup_epochs,
        args.scheduler_patience, factor=0.5,
        cosine_tmax=args.cosine_tmax, eta_min=args.eta_min)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=4096.0)
    loss_function = nn.MSELoss()
    results_root = Path(args.results_root) if args.results_root else Path("results")
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
        "loss": "MSE",
        "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.lr, "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.grad_clip_norm,
        "scheduler": args.scheduler, "warmup_epochs": args.warmup_epochs,
        "cosine_tmax": args.cosine_tmax, "eta_min": args.eta_min,
        "scheduler_patience": args.scheduler_patience,
        "early_stopping_patience": args.early_stopping_patience,
        "top_k": args.top_k,
        "split_sizes": sizes,
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
              "ci", "r2", "rm2", "mae", "lr", "max_grad", "amp_overflows",
              "seconds"]
    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=fields).writeheader()
    top_k, best_mse, stale_epochs = [], float("inf"), 0
    epoch_times, total_overflows = [], 0
    best_validation_metrics = None
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, max_grad, overflows = train_epoch(
            model, train_loader, optimizer, loss_function, scaler, device, epoch)
        y_true, y_pred = predict(model, validation_loader, device)
        validation_mse = float(np.mean((y_true - y_pred) ** 2))
        if scheduler_kind == "plateau":
            scheduler.step(validation_mse)
        else:
            scheduler.step()
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
            "lr": optimizer.param_groups[0]["lr"],
            "max_grad": max_grad, "amp_overflows": overflows,
            "seconds": round(seconds, 2),
        }
        with history_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=fields).writerow(row)
        print(
            f"Epoch {epoch}: val MSE={validation_mse:.6f}, best={best_mse:.6f}, "
            f"lr={row['lr']:.2e}, grad={max_grad:.2f}, overflow={overflows}, "
            f"{seconds:.1f}s",
            flush=True)
        if stale_epochs >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}.", flush=True)
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
    print(json.dumps(test_results, indent=2), flush=True)
    print(f"Results saved to {run_dir}", flush=True)


if __name__ == "__main__":
    main()

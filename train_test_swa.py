"""train_test_swa.py - cosine (short T_max) + SWA, no early stopping, plain MSE.

Same clean recipe as train_test.py (plain MSE, AdamW weight decay 1e-4, cosine
T_max 300 / eta_min 1e-5 / warmup 5) but with early stopping replaced by
stochastic weight averaging (SWA, ``torch.optim.swa_utils``) over the tail.
The margin / floor / label-noise mechanisms are removed from the recipe
entirely (see train_test.py docstring for why).

Example:
  python train_test_swa.py --model MDS_dta
  python train_test_swa.py --model MDS_dta --swa-start 200 --epochs 350
"""

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import train_test as tt


SEED = 42
DEVICE = "cuda:0"
EPOCHS = 400
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 5.0
COSINE_TMAX = 300
ETA_MIN = 1e-5
WARMUP_EPOCHS = 5
SWA_START = 225
TOP_K = 3
NUM_WORKERS = 4


def main():
    parser = argparse.ArgumentParser(
        description="Cosine (short T_max) + SWA without early stopping (plain MSE).")
    parser.add_argument("--dataset", default="davis")
    parser.add_argument("--model", default="MDS_dta")
    parser.add_argument("--model-params", default=None,
                        help="Optional JSON object overriding model hyper-parameters.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip-norm", type=float, default=GRAD_CLIP_NORM)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--cosine-tmax", type=int, default=COSINE_TMAX)
    parser.add_argument("--eta-min", type=float, default=ETA_MIN)
    parser.add_argument("--swa-start", type=int, default=SWA_START)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    dataset_name = args.dataset.strip().lower()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    tt.seed_everything(args.seed)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model, model_module_name, model_class_name, applied_params, requested_params = \
        tt.build_model(args.model, args.model_params)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Dataset: {dataset_name} | Model: {model_module_name} "
          f"({model_class_name}) | Device: {device}")
    print(f"Applied model params: {applied_params}")
    print(f"Trainable parameters: {n_params:,} ({n_params / 1e6:.2f} M)")

    train_loader, validation_loader, test_loader, sizes, split_file = \
        tt.make_loaders(device, dataset_name, args.batch_size, args.eval_batch_size)
    print(f"Fixed split train/validation/test: {sizes}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler, scheduler_kind = tt.build_scheduler(
        optimizer, "cosine", args.epochs, args.warmup_epochs,
        tt.SCHEDULER_PATIENCE, factor=0.5,
        cosine_tmax=args.cosine_tmax, eta_min=args.eta_min)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=4096.0)
    loss_function = nn.MSELoss()

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_model.eval()

    run_dir = Path("results") / time.strftime(
        f"{model_class_name}_{dataset_name}_swa_%Y%m%d-%H%M%S")
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
        "scheduler": "cosine", "warmup_epochs": args.warmup_epochs,
        "cosine_tmax": args.cosine_tmax, "eta_min": args.eta_min,
        "swa_start": args.swa_start, "top_k": args.top_k,
        "early_stopping": "disabled",
        "split_sizes": sizes,
        "split_protocol": "one fixed 4/1/1 split of six parts; no fold rotation",
    }
    (run_dir / "config.json").write_text(
        json.dumps(saved_config, indent=2), encoding="utf-8")
    shutil.copyfile(split_file, run_dir / "split_indices.json")

    fields = ["epoch", "train_loss", "mse", "rmse", "pearson", "spearman",
              "ci", "r2", "rm2", "mae", "lr", "swa_val_mse", "max_grad",
              "amp_overflows", "seconds"]
    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=fields).writeheader()

    top_k, best_mse = [], float("inf")
    epoch_times, total_overflows = [], 0
    best_validation_metrics = None
    best_swa_val_mse = None

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, max_grad, overflows = tt.train_epoch(
            model, train_loader, optimizer, loss_function, scaler, device, epoch)
        y_true, y_pred = tt.predict(model, validation_loader, device)
        validation_mse = float(np.mean((y_true - y_pred) ** 2))

        scheduler.step()
        tt.retain_top_k(top_k, model, run_dir, epoch, validation_mse)

        improved = validation_mse < best_mse
        if improved:
            best_mse = validation_mse
            best_validation_metrics = tt.metrics(y_true, y_pred)

        swa_val_mse = ""
        if epoch >= args.swa_start:
            swa_model.update_parameters(model)
            sy_true, sy_pred = tt.predict(swa_model, validation_loader, device)
            swa_val_mse = float(np.mean((sy_true - sy_pred) ** 2))
            if best_swa_val_mse is None or swa_val_mse < best_swa_val_mse:
                best_swa_val_mse = swa_val_mse

        seconds = time.time() - start
        epoch_times.append(seconds)
        total_overflows += overflows
        detailed = best_validation_metrics if improved else {}
        row = {
            "epoch": epoch, "train_loss": train_loss, "mse": validation_mse,
            **{name: detailed.get(name, "") for name in
               ("rmse", "pearson", "spearman", "ci", "r2", "rm2", "mae")},
            "lr": optimizer.param_groups[0]["lr"],
            "swa_val_mse": swa_val_mse,
            "max_grad": max_grad, "amp_overflows": overflows,
            "seconds": round(seconds, 2),
        }
        with history_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=fields).writerow(row)
        print(
            f"Epoch {epoch}: val MSE={validation_mse:.6f}, best={best_mse:.6f}, "
            f"swa={swa_val_mse if swa_val_mse == '' else round(swa_val_mse, 6)}, "
            f"lr={row['lr']:.2e}, grad={max_grad:.2f}, {seconds:.1f}s",
            flush=True)

    summary = {
        "dataset": dataset_name,
        "model_class": model_class_name,
        "epochs_completed": len(epoch_times),
        "best_epoch": top_k[0]["epoch"],
        "best_validation_metrics": best_validation_metrics,
        "best_swa_val_mse": best_swa_val_mse,
        "top3_validation_mse": [
            {"epoch": item["epoch"], "mse": item["mse"]} for item in top_k],
        "mean_epoch_seconds": float(np.mean(epoch_times)),
        "total_amp_overflows": total_overflows,
    }
    (run_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    best_checkpoint = torch.load(top_k[0]["path"], map_location="cpu", weights_only=True)
    best_result, labels, best_predictions = tt.test_model(
        model, best_checkpoint["model_state_dict"], test_loader, device)
    top3_result, _, top3_predictions = tt.test_model(
        model, tt.averaged_state(top_k), test_loader, device)
    swa_result, _, swa_predictions = tt.test_model(
        swa_model, swa_model.state_dict(), test_loader, device)
    best_result["selected_epoch"] = top_k[0]["epoch"]
    top3_result["averaged_epochs"] = [item["epoch"] for item in top_k]
    swa_result["swa_start"] = args.swa_start
    test_results = {
        "best_checkpoint": best_result,
        "top3_average": top3_result,
        "swa": swa_result,
    }
    (run_dir / "test_metrics.json").write_text(
        json.dumps(test_results, indent=2), encoding="utf-8")
    np.savez_compressed(
        run_dir / "test_predictions.npz", labels=labels,
        best_checkpoint=best_predictions, top3_average=top3_predictions,
        swa=swa_predictions)
    print(json.dumps(test_results, indent=2), flush=True)
    print(f"Results saved to {run_dir}", flush=True)


if __name__ == "__main__":
    main()


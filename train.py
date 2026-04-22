import csv
import json
import os
import random
import time

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from tqdm import tqdm

from models.MDS_DTA import MDSDTA

from utils import TestbedDataset, rmse, mse, pearson, spearman, ci, r2, mae, rm2

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# User-editable settings
DATASET_NAME = 'bindingdb'  # options: 'davis','kiba','bindingdb'
MODEL = MDSDTA
DEFAULT_CUDA = 0
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 1e-4
NUM_EPOCHS = 1000
LOG_INTERVAL = 100
WEIGHT_DECAY = 0
GRAD_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 100
USE_MIXED_PRECISION = True
SEED = 42
OUTPUT_DIR = 'results'
NUM_WORKERS = 4
PERSISTENT_WORKERS = True
PIN_MEMORY = True
ENABLE_CUDNN_BENCHMARK = False


# Utilities
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cuda_index: int = DEFAULT_CUDA) -> torch.device:
    return torch.device(f'cuda:{cuda_index}' if torch.cuda.is_available() else 'cpu')


def instantiate_model(model_cls, device: torch.device):
    return model_cls().to(device)


def make_run_dir(output_base: str, model_name: str, dataset: str):
    os.makedirs(output_base, exist_ok=True)
    prefix = f"{model_name}_{dataset}_run"
    existing = [d for d in os.listdir(output_base) if d.startswith(prefix)]
    next_idx = max([int(d.split('_run')[-1].split('_')[0]) for d in existing] or [0]) + 1
    ts = time.strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(output_base, f"{prefix}{next_idx:03d}_{ts}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


# Training / evaluation loops
def train_epoch(model, device, loader, optimizer, loss_fn, scaler, epoch=0):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train E{epoch} LR={optimizer.param_groups[0]['lr']:.3e}")
    for batch_idx, data in pbar:
        data = data.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            output = model(data)
            loss = loss_fn(output, data.y.view(-1, 1).float())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        if batch_idx % LOG_INTERVAL == 0:
            pbar.set_postfix(loss=loss.item(), lr=f"{optimizer.param_groups[0]['lr']:.3e}")

    avg_loss = running_loss / max(1, len(loader))
    return avg_loss


def predict(model, device, loader):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for data in tqdm(loader, desc="Predict"):
            data = data.to(device, non_blocking=True)
            out = model(data)
            preds.append(out.detach().cpu())
            trues.append(data.y.view(-1, 1).detach().cpu())
    preds = torch.cat(preds, dim=0).numpy().flatten()
    trues = torch.cat(trues, dim=0).numpy().flatten()
    return trues, preds


def append_csv_row(csv_file, row):
    with open(csv_file, 'a', newline='') as f:
        csv.writer(f).writerow(row)


def compute_metrics(g, p):
    return {
        'pearson': pearson(g, p),
        'spearman': spearman(g, p),
        'ci': ci(g, p),
        'r2': r2(g, p),
        'rm2': rm2(g, p),
        'mae': mae(g, p)
    }


# Main
def main():
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    if ENABLE_CUDNN_BENCHMARK and device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print('Enabled torch.backends.cudnn.benchmark for possible speedup')

    dataset = DATASET_NAME.lower()
    train_data = TestbedDataset(root='data', dataset=dataset+ '_train')
    test_data = TestbedDataset(root='data', dataset=dataset + '_test')

    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False)

    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                             persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False)

    model = instantiate_model(MODEL, device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500, verbose=True
    )

    amp_enabled = USE_MIXED_PRECISION and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if amp_enabled:
        print('Using mixed precision training (AMP).')

    model_name = MODEL.__name__
    run_dir = make_run_dir(OUTPUT_DIR, model_name, dataset)
    print(f"Saving this run's outputs to: {run_dir}")

    history_csv = os.path.join(run_dir, f'results_{model_name}_{dataset}.csv')
    best_ckpt = os.path.join(run_dir, f'best_{model_name}_{dataset}.pt')
    last_ckpt = os.path.join(run_dir, f'last_{model_name}_{dataset}.pt')
    best_model_pth = os.path.join(run_dir, 'best_model.pth')
    train_losses_json = os.path.join(run_dir, 'train_losses.json')

    config = dict(
        dataset=DATASET_NAME,
        model=model_name,
        train_batch_size=TRAIN_BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
        lr=LR,
        num_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        grad_clip_norm=GRAD_CLIP_NORM,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        use_mixed_precision=USE_MIXED_PRECISION,
        seed=SEED,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=PIN_MEMORY,
        cudnn_benchmark=ENABLE_CUDNN_BENCHMARK,
        lr_scheduler='ReduceLROnPlateau',
        scheduler_factor=0.5,
        scheduler_patience=75,
    )
    with open(os.path.join(run_dir, 'config.json'), 'w') as fh:
        json.dump(config, fh, indent=2)

    metric_names = ['epoch', 'rmse', 'mse', 'pearson', 'spearman', 'ci', 'r2', 'rm2', 'mae', 'train_loss',
                    'epoch_time_s']

    best_mse = float('inf')
    best_epoch = -1
    no_improve_cnt = 0
    train_losses = []
    best_r2 = None
    best_rm2 = None

    with open(history_csv, 'w', newline='') as f:
        csv.writer(f).writerow(metric_names)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, device, train_loader, optimizer, loss_fn, scaler, epoch)
        g, p = predict(model, device, test_loader)

        current_rmse = rmse(g, p)
        current_mse = mse(g, p)

        # 根据验证MSE调整学习率
        scheduler.step(current_mse)

        epoch_time = time.time() - epoch_start

        improved = current_mse < best_mse
        should_record = (epoch % 100 == 0) or improved
        if should_record:
            m = compute_metrics(g, p)
            row = [
                epoch,
                current_rmse,
                current_mse,
                m['pearson'],
                m['spearman'],
                m['ci'],
                m['r2'],
                m['rm2'],
                m['mae'],
                train_loss,
                f"{epoch_time:.2f}",
            ]
            append_csv_row(history_csv, row)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'mse': current_mse}, last_ckpt)

        train_losses.append(train_loss)

        if improved:
            best_mse = current_mse
            best_epoch = epoch
            no_improve_cnt = 0
            best_r2 = m['r2'] if should_record else best_r2
            best_rm2 = m['rm2'] if should_record else best_rm2
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'mse': best_mse}, best_ckpt)
            torch.save(model.state_dict(), best_model_pth)
            print(f"Epoch {epoch}: New best MSE={best_mse:.6f} — checkpoint saved to {best_ckpt} — LR={optimizer.param_groups[0]['lr']:.3e}")
        else:
            no_improve_cnt += 1
            print(f"Epoch {epoch}: MSE={current_mse:.6f} (best {best_mse:.6f} at epoch {best_epoch}) — LR={optimizer.param_groups[0]['lr']:.3e}")

        if no_improve_cnt >= EARLY_STOPPING_PATIENCE:
            print(f"No improvement in {EARLY_STOPPING_PATIENCE} epochs — early stopping at epoch {epoch}.")
            break

    print(f"Training finished. Best MSE={best_mse:.6f} at epoch {best_epoch}.")

    with open(train_losses_json, 'w') as fh:
        json.dump({'train_losses': train_losses}, fh)

    summary = {
        'run_dir': run_dir,
        'best_mse': float(best_mse) if best_mse is not None else None,
        'best_epoch': int(best_epoch) if best_epoch is not None else None,
        'best_r2': float(best_r2) if best_r2 is not None else None,
        'best_rm2': float(best_rm2) if best_rm2 is not None else None,
    }
    with open(os.path.join(run_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f"All artifacts written to {run_dir}")


if __name__ == '__main__':
    main()

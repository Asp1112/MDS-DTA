"""Cold-start training entry (entity-level six-fold manifests).

Supports the MDS CombinedDTA-family models and the three public baselines:
  * DeepDTA      --model deepdta
  * GraphDTA     --model graphdta_gcn|graphdta_gat|graphdta_gat_gcn|
                          graphdta_ginconv
  * DeepDTAGen   --model deepdtagen
  * MDS family   --model MDS_dta|MDS_dta|MDS_dta|
                          ... (any module under models/)

Example:
  python train_cold.py --dataset davis --setting cold_drug --fold 0 \
      --model deepdta
  python train_cold.py --dataset davis --setting cold_both --fold 0 \
      --model graphdta_gcn
  python train_cold.py --dataset kiba --setting cold_target --fold 0 \
      --model deepdtagen
  python train_cold.py --dataset bindingdb --setting cold_drug --fold 0 \
      --model MDS_dta
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset


COMMON = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "common"
sys.path.insert(0, str(COMMON))

import exp_common  # noqa: E402
from exp_common import (  # noqa: E402
    EXPERIMENTS_ROOT, MDS_ROOT, SavedGraphDataset, add_experiment_args,
    build_model, check_done, load_manifest, load_rows, make_run_dir,
    metrics, run_training, seed_everything,
)


EXPERIMENT = "cold_start"
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = ["cold_drug", "cold_target", "cold_both"]
GRAPH_VARIANTS = ["graphdta_gcn", "graphdta_gat", "graphdta_gat_gcn",
                  "graphdta_ginconv"]

FAMILY_DEFAULTS = {
    "deepdta": {"batch_size": 256, "eval_batch_size": 256, "amp": False,
                "optimizer": "adamw", "weight_decay": 1e-4},
    "graphdta": {"batch_size": 256, "eval_batch_size": 256, "amp": False,
                 "optimizer": "adamw", "weight_decay": 1e-4},
    "deepdtagen": {"batch_size": 32, "eval_batch_size": 128, "amp": False,
                   "optimizer": "adam", "weight_decay": 0.0},
    "mds": {"batch_size": 256, "eval_batch_size": 256, "amp": True,
            "optimizer": "adamw", "weight_decay": 1e-4},
}


def family_of(model_spec):
    if model_spec == "deepdta":
        return "deepdta"
    if model_spec in GRAPH_VARIANTS:
        return "graphdta"
    if model_spec == "deepdtagen":
        return "deepdtagen"
    return "mds"


def build_deepdta_loaders(dataset, split, batch_size, eval_batch_size,
                          workers, device):
    from baselines.deepdta_baseline import label_sequence, label_smiles
    df = load_rows(dataset)
    def encode(indices):
        drugs = np.stack([label_smiles(s)
                          for s in df["compound_iso_smiles"].iloc[indices]])
        prots = np.stack([label_sequence(s)
                          for s in df["target_sequence"].iloc[indices]])
        y = df["affinity"].iloc[indices].to_numpy(dtype=np.float32)
        return (torch.from_numpy(drugs), torch.from_numpy(prots),
                torch.from_numpy(y))
    tr_x, tr_p, tr_y = encode(split["train_indices"])
    va_x, va_p, va_y = encode(split["validation_indices"])
    te_x, te_p, te_y = encode(split["test_indices"])
    options = {"num_workers": workers, "pin_memory": device.type == "cuda"}
    train = TorchDataLoader(TensorDataset(tr_x, tr_p, tr_y),
                            batch_size=batch_size, shuffle=True, **options)
    validation = TorchDataLoader(TensorDataset(va_x, va_p, va_y),
                                 batch_size=eval_batch_size, shuffle=False,
                                 **options)
    test = TorchDataLoader(TensorDataset(te_x, te_p, te_y),
                           batch_size=eval_batch_size, shuffle=False, **options)
    return train, validation, test


def build_pyg_loaders(dataset, split, batch_size, eval_batch_size,
                      workers, device, kind):
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from exp_common import IndexedSubset
    if kind == "graphdta":
        path = MDS_ROOT / "data" / "baselines" / f"{dataset}_graphdta_all.pt"
    else:
        path = MDS_ROOT / "data" / "baselines" / f"{dataset}_deepdtagen_all.pt"
    if not path.exists():
        raise SystemExit(f"Missing baseline data file: {path}")
    base = SavedGraphDataset(str(path))
    options = {"num_workers": workers, "pin_memory": device.type == "cuda",
               "persistent_workers": workers > 0}
    train = PyGDataLoader(IndexedSubset(base, split["train_indices"]),
                          batch_size=batch_size, shuffle=True, **options)
    validation = PyGDataLoader(IndexedSubset(base, split["validation_indices"]),
                               batch_size=eval_batch_size, shuffle=False,
                               **options)
    test = PyGDataLoader(IndexedSubset(base, split["test_indices"]),
                         batch_size=eval_batch_size, shuffle=False, **options)
    return train, validation, test


def build_mds_loaders(dataset, split, batch_size, eval_batch_size,
                      workers, device):
    from exp_common import loaders_from_indices
    return loaders_from_indices(dataset, [
        split["train_indices"], split["validation_indices"],
        split["test_indices"]], batch_size, eval_batch_size, workers, device)


def forward_loss_mds(model, batch, loss_fn, device):
    batch = batch.to(device, non_blocking=True)
    output = model(batch)
    target = batch.y.view(-1, 1).float()
    return loss_fn(output, target), batch.y.view(-1), output


def predict_mds(model, loader, device):
    model.eval()
    labels, preds = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            labels.append(batch.y.view(-1).cpu().numpy())
            preds.append(model(batch).view(-1).cpu().numpy())
    return np.concatenate(labels), np.concatenate(preds)


def forward_loss_deepdta(model, batch, loss_fn, device):
    drug, protein, y = batch
    drug, protein = drug.to(device, non_blocking=True), \
        protein.to(device, non_blocking=True)
    target = y.to(device, non_blocking=True).view(-1, 1).float()
    output = model(drug, protein)
    return loss_fn(output, target), y.view(-1), output


def predict_deepdta(model, loader, device):
    model.eval()
    labels, preds = [], []
    with torch.inference_mode():
        for drug, protein, y in loader:
            drug, protein = drug.to(device, non_blocking=True), \
                protein.to(device, non_blocking=True)
            labels.append(y.numpy())
            preds.append(model(drug, protein).view(-1).cpu().numpy())
    return np.concatenate(labels), np.concatenate(preds)


def forward_loss_deepdtagen(model, batch, loss_fn, device):
    batch = batch.to(device, non_blocking=True)
    prediction, _, lm_loss, kl_loss = model(batch)
    target = batch.y.view(-1, 1).float()
    mse_loss = loss_fn(prediction, target)
    loss = kl_loss * 0.001 + mse_loss + lm_loss
    return loss, batch.y.view(-1), prediction


def predict_deepdtagen(model, loader, device):
    model.eval()
    labels, preds = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            prediction, _, _, _ = model(batch)
            labels.append(batch.y.view(-1).cpu().numpy())
            preds.append(prediction.view(-1).cpu().numpy())
    return np.concatenate(labels), np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser(
        description="Train cold-start models on entity-level six-fold splits.")
    parser = add_experiment_args(parser, EXPERIMENT)
    parser.add_argument("--setting", choices=SETTINGS, default="cold_drug")
    parser.add_argument("--fold", type=int, required=True, choices=range(6))
    parser.add_argument("--model", default="MDS_dta",
                        help="deepdta | graphdta_gcn | graphdta_gat | "
                             "graphdta_gat_gcn | graphdta_ginconv | "
                             "deepdtagen | <models.* module>")
    parser.add_argument("--model-params", default=None)
    args = parser.parse_args()
    dataset = args.dataset.strip().lower()
    setting = args.setting
    fold = args.fold
    model_spec = args.model
    family = family_of(model_spec)
    defaults = FAMILY_DEFAULTS[family]
    manifest_path = HERE / "data" / "splits" / dataset / setting / \
        f"fold_{fold}.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"Missing split manifest: {manifest_path} "
            "(run prepare_cold_start.py first).")
    split = load_manifest(manifest_path)
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    epochs = 2 if args.dry else args.epochs
    batch_size = args.batch_size or defaults["batch_size"]
    if args.dry and args.batch_size is None:
        batch_size = min(64, batch_size)
    eval_batch_size = args.eval_batch_size or defaults["eval_batch_size"]
    if args.dry:
        eval_batch_size = min(128, eval_batch_size)
    workers = 0 if args.dry else args.workers
    args.batch_size = batch_size
    args.eval_batch_size = eval_batch_size
    args.epochs = epochs
    if family == "deepdta":
        from baselines.deepdta_baseline import DeepDTAModel
        model = DeepDTAModel().to(device)
        train_loader, validation_loader, test_loader = build_deepdta_loaders(
            dataset, split, batch_size, eval_batch_size, workers, device)
        forward_loss, predict_fn = forward_loss_deepdta, predict_deepdta
        model_label = "DeepDTA"
        model_module, model_class = "baselines.deepdta_baseline", "DeepDTA"
        applied = {"embed_dim": 128, "num_filters": 32, "dropout": 0.1}
    elif family == "graphdta":
        from baselines.graphdta_baseline import MODELS, VARIANTS
        model_class = VARIANTS[model_spec]
        model = MODELS[model_class]().to(device)
        train_loader, validation_loader, test_loader = build_pyg_loaders(
            dataset, split, batch_size, eval_batch_size, workers, device,
            "graphdta")
        forward_loss, predict_fn = forward_loss_mds, predict_mds
        model_label = model_class
        model_module = "baselines.graphdta_baseline"
        applied = {"variant": model_spec}
    elif family == "deepdtagen":
        official = MDS_ROOT / "baselines" / "deepdtagen_official"
        sys.path.insert(0, str(official))
        import pickle
        from FetterGrad import FetterGrad
        from model import DeepDTAGen
        from utils import Tokenizer
        tokenizer_path = MDS_ROOT / "data" / "baselines" / \
            f"{dataset}_tokenizer.pkl"
        if not tokenizer_path.exists():
            raise SystemExit(f"Missing tokenizer: {tokenizer_path}")
        with open(tokenizer_path, "rb") as fh:
            tokenizer = pickle.load(fh)
        model = DeepDTAGen(tokenizer).to(device)
        train_loader, validation_loader, test_loader = build_pyg_loaders(
            dataset, split, batch_size, eval_batch_size, workers, device,
            "deepdtagen")
        forward_loss, predict_fn = forward_loss_deepdtagen, predict_deepdtagen
        model_label = "DeepDTAGen"
        model_module, model_class = "baselines.deepdtagen_baseline", "DeepDTAGen"
        applied = {"kl_weight": 0.001}
    else:
        model, model_module, model_class, applied, requested = build_model(
            model_spec, args.model_params)
        model = model.to(device)
        train_loader, validation_loader, test_loader = build_mds_loaders(
            dataset, split, batch_size, eval_batch_size, workers, device)
        forward_loss, predict_fn = forward_loss_mds, predict_mds
        model_label = model_class
        requested = args.model_params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Cold start | dataset={dataset} setting={setting} fold={fold} "
          f"family={family} model={model_spec} device={device} "
          f"params={n_params:,}", flush=True)
    print(f"Split sizes train/val/test: {split['sizes']}", flush=True)
    if family == "deepdtagen":
        from FetterGrad import FetterGrad
        optimizer = FetterGrad(torch.optim.Adam(
            model.parameters(), lr=args.lr))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer.optimizer, mode="min", factor=0.5,
            patience=args.scheduler_patience)
    else:
        if defaults["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=args.scheduler_patience)
    loss_fn = nn.MSELoss()
    prefix = f"{model_label}_{dataset}_{setting}_fold{fold}"
    results_root = args.results_root
    if args.skip_done and results_root and check_done(results_root, prefix):
        print(f"[skip] {prefix}: completed run already exists.", flush=True)
        return
    run_training(
        args, EXPERIMENT, prefix, split, list(split["sizes"].values()),
        model, device, train_loader, validation_loader, test_loader,
        optimizer, scheduler, loss_fn, forward_loss, predict_fn,
        model_module, model_class, applied,
        args.model_params if family == "mds" else applied,
        {"family": family, "amp": defaults["amp"], "setting": setting,
         "fold": fold},
        manifest_path)


if __name__ == "__main__":
    main()

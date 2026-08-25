from __future__ import annotations

import copy
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from crc64iso.crc64iso import crc64


MDS_ROOT = Path(os.environ.get("MDS_WORK_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = Path(os.environ.get("MDS_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
OUT = ROOT / "task_dataset_1067"
RUNTIME = OUT / "runtime"
CAND_CSV = ROOT / "Supplementary_Data_1_candidate_library_metadata_10026.csv"
SCORE_CSV = ROOT / "打分结果.csv"

SELECTED_ACCESSIONS = [
    "Q72X44", "P40353", "O31995", "Q66165", "P48026",
    "Q5NHR0", "D2Z028", "Q8P051", "Q9KL03", "P26825",
]
PAP_NAME = "4-aminophenol"
PAP_SMILES = "C1=CC(=CC=C1N)O"
SEED = 42
MAX_LEN = 1000

sys.path.insert(0, str(MDS_ROOT))
from models.combined_dta import CombinedDTA  # noqa: E402
from utils import TestbedDataset  # noqa: E402
from run_1067_rank_selected import seq_to_array, smile_graph, load_initial_state  # noqa: E402
from run_v1_train_eval_top100 import make_dataset, predict, selected_crc_map, norm_seq  # noqa: E402


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(OUT / "mds_pAAP_1067.csv", dtype=str)
    df["Label"] = df["Label"].astype(int)
    df["Sequence"] = df["Sequence"].map(norm_seq)
    task_dataset = make_dataset(df, "task1067_v1_multitask_task", RUNTIME)

    cand = pd.read_csv(CAND_CSV, dtype=str)
    cand_crc = set(cand["sequence_crc64"].str.upper().str.strip())
    sel_map = selected_crc_map(cand)
    sel_crc = set(sel_map.values())

    score = pd.read_csv(SCORE_CSV, dtype=str)
    score["y_pred"] = pd.to_numeric(score["y_pred"], errors="coerce")
    score["Sequence"] = score["protein_sequence"].map(norm_seq)
    score["crc64"] = [crc64(s).upper() for s in score["Sequence"]]
    dist_df = score[score["crc64"].isin(cand_crc)].drop_duplicates("crc64", keep="first").copy()
    dist_df = dist_df[~dist_df["crc64"].isin(sel_crc)].reset_index(drop=True)
    dist_df["Label"] = dist_df["y_pred"]
    dist_df["Ligand_SMILES"] = PAP_SMILES
    dist_dataset = make_dataset(dist_df, "task1067_v1_multitask_distill", RUNTIME)

    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.15,
        random_state=SEED,
        stratify=df["Label"],
    )
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    task_train_loader = DataLoader(Subset(task_dataset, train_idx.tolist()), batch_size=32, shuffle=True, num_workers=0)
    task_val_loader = DataLoader(Subset(task_dataset, val_idx.tolist()), batch_size=64, shuffle=False, num_workers=0)
    dist_loader = DataLoader(dist_dataset, batch_size=64, shuffle=True, num_workers=0)

    init_path = MDS_ROOT / "models" / "best_model_pAAP_y.pth"
    model = CombinedDTA(drug_atom_feat_dim=78).to(device)
    load_result = load_initial_state(model, init_path, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    patience = 10
    history = []
    start = time.time()
    max_epochs = 40
    distill_weight = 50.0

    for epoch in range(1, max_epochs + 1):
        model.train()
        task_losses = []
        dist_losses = []

        for batch in task_train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                output = model(batch)
                loss = loss_fn(output, batch.y.view(-1, 1).float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            task_losses.append(float(loss.detach().cpu()))

        for batch in dist_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                output = model(batch)
                loss = loss_fn(output, batch.y.view(-1, 1).float()) * distill_weight
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            dist_losses.append(float(loss.detach().cpu()))

        yv, pv = [], []
        model.eval()
        with torch.no_grad():
            for batch in task_val_loader:
                batch = batch.to(device)
                pred = model(batch).view(-1)
                yv.extend(batch.y.view(-1).cpu().numpy().tolist())
                pv.extend(pred.cpu().numpy().tolist())
        yv = np.asarray(yv)
        pv = np.asarray(pv)
        val_mse = float(np.mean((yv - pv) ** 2))
        history.append(
            {
                "epoch": epoch,
                "task_train_mse": float(np.mean(task_losses)),
                "distill_train_mse": float(np.mean(dist_losses)),
                "task_validation_mse": val_mse,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if val_mse < best_val - 1e-8:
            best_val = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale >= patience:
            print(
                f"epoch={epoch} task={np.mean(task_losses):.6f} dist={np.mean(dist_losses):.6f} "
                f"val={val_mse:.6f} best={best_val:.6f} stale={stale}",
                flush=True,
            )
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("No best state recorded")
    model.load_state_dict(best_state, strict=True)
    model.to(device)

    eval_df = score[score["crc64"].isin(cand_crc)].drop_duplicates("crc64", keep="first").copy()
    eval_df["Label"] = 0
    eval_df["Ligand_SMILES"] = PAP_SMILES
    eval_dataset = make_dataset(eval_df, "task1067_v1_multitask_eval10026", RUNTIME)
    eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False, num_workers=0)
    eval_scores = predict(model, eval_loader, device)
    eval_df["new_y_pred"] = eval_scores
    eval_df["new_rank"] = eval_df["new_y_pred"].rank(ascending=False, method="first").astype(int)

    selected_rows = []
    for acc, crc in sel_map.items():
        row = eval_df[eval_df["crc64"] == crc]
        selected_rows.append(
            {
                "accession": acc,
                "rank": int(row["new_rank"].iloc[0]) if not row.empty else None,
                "y_pred": float(row["new_y_pred"].iloc[0]) if not row.empty else None,
            }
        )
    sel_df = pd.DataFrame(selected_rows).sort_values("rank", key=lambda s: pd.to_numeric(s, errors="coerce"))
    sel_df["in_top100"] = sel_df["rank"].notna() & (sel_df["rank"] <= 100)

    model_path = OUT / "best_model_1067_v1_multitask.pth"
    torch.save(best_state, model_path)
    pd.DataFrame(history).to_csv(OUT / "training_history_1067_v1_multitask.csv", index=False)
    sel_df.to_csv(OUT / "selected10_top100_v1_multitask.csv", index=False, encoding="utf-8-sig")

    meta = {
        "method": "multitask: 1067 task labels + 10026 screening-score distillation",
        "initialization": str(init_path),
        "device": str(device),
        "train_n": int(len(train_idx)),
        "validation_n": int(len(val_idx)),
        "distill_n": int(len(dist_df)),
        "distill_weight": distill_weight,
        "best_epoch": best_epoch,
        "best_task_validation_mse": best_val,
        "epochs_run": len(history),
        "elapsed_seconds": time.time() - start,
        "selected10_in_top100": bool(sel_df["in_top100"].all()),
        "selected10_count_in_top100": int(sel_df["in_top100"].sum()),
        "selected10": sel_df.to_dict(orient="records"),
    }
    with (OUT / "selected10_top100_v1_multitask_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(sel_df.to_string(index=False))


if __name__ == "__main__":
    main()

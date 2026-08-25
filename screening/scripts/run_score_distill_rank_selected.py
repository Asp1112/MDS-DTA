from __future__ import annotations

import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader


MDS_ROOT = Path(os.environ.get("MDS_WORK_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = Path(os.environ.get("MDS_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
OUT = ROOT / "task_dataset_1067"
RUNTIME = OUT / "runtime"
SCORE_CSV = ROOT / "打分结果.csv"
CAND_CSV = ROOT / "Supplementary_Data_1_candidate_library_metadata_10026.csv"
SELECTED_CSV = ROOT / "docking_comparison" / "09_corrected_screen" / "corrected_rule_application.csv"
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
from crc64iso.crc64iso import crc64


def make_dataset(df: pd.DataFrame, name: str, runtime: Path, value_col: str = "y_pred") -> TestbedDataset:
    xd = np.asarray([PAP_SMILES] * len(df), dtype=object)
    xt = np.asarray([seq_to_array(s) for s in df["Sequence"]])
    y = df[value_col].astype(float).to_numpy()
    graph = {PAP_SMILES: smile_graph(PAP_SMILES)}
    return TestbedDataset(
        root=str(runtime),
        dataset=name,
        xd=xd,
        xt=xt,
        y=y,
        smile_graph=graph,
        force_reprocess=True,
    )


def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).view(-1)
            ys.extend(batch.y.view(-1).cpu().numpy().tolist())
            ps.extend(pred.cpu().numpy().tolist())
    return np.asarray(ys), np.asarray(ps)


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

    cand = pd.read_csv(CAND_CSV, dtype=str)
    cand_crc = set(cand["sequence_crc64"].str.upper().str.strip())

    selected = pd.read_csv(SELECTED_CSV, dtype=str)
    selected = selected[selected["accession"].isin(SELECTED_ACCESSIONS)].copy()
    selected = selected.sort_values("rank", key=lambda s: s.astype(int)).reset_index(drop=True)
    selected["old_y_pred"] = pd.to_numeric(selected["y_pred"], errors="coerce")
    selected["Sequence"] = selected["sequence"].str.strip().str.upper()
    selected["crc64"] = [crc64(s).upper() for s in selected["Sequence"]]
    selected_crc = set(selected["crc64"])
    selected_seq = set(selected["Sequence"])

    score = pd.read_csv(SCORE_CSV, dtype=str)
    score["y_pred"] = pd.to_numeric(score["y_pred"], errors="coerce")
    score["Sequence"] = score["protein_sequence"].str.strip().str.upper()
    score["crc64"] = [crc64(s).upper() for s in score["Sequence"]]
    score = score[score["crc64"].isin(cand_crc)].copy()
    score = score[
        ~score["crc64"].isin(selected_crc)
        & ~score["Sequence"].isin(selected_seq)
    ].copy()
    score = score.drop_duplicates(subset=["Sequence"], keep="first").reset_index(drop=True)
    print("training candidate records after masking selected", len(score), flush=True)

    score["Label"] = 0
    dataset = make_dataset(score, "task1067_score_distill", RUNTIME, value_col="y_pred")

    strat_col = pd.qcut(score["y_pred"], q=5, labels=False, duplicates="drop")
    train_idx, val_idx = train_test_split(
        np.arange(len(score)),
        test_size=0.10,
        random_state=SEED,
        stratify=strat_col,
    )
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=128, shuffle=False, num_workers=0)

    init_path = MDS_ROOT / "models" / "best_model_pAAP_y.pth"
    model = CombinedDTA(drug_atom_feat_dim=78).to(device)
    load_result = load_initial_state(model, init_path, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=12)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    patience = 25
    history = []
    start = time.time()
    max_epochs = 120

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
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
            losses.append(float(loss.detach().cpu()))

        yv, pv = predict(model, val_loader, device)
        val_mse = float(np.mean((yv - pv) ** 2))
        scheduler.step(val_mse)
        history.append(
            {
                "epoch": epoch,
                "train_mse": float(np.mean(losses)),
                "validation_mse": val_mse,
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
                f"epoch={epoch} train_mse={np.mean(losses):.6f} val_mse={val_mse:.6f} best={best_val:.6f} stale={stale}",
                flush=True,
            )
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("No best state recorded")
    model.load_state_dict(best_state, strict=True)
    model.to(device)

    selected["Label"] = 0
    cand_dataset = make_dataset(selected.rename(columns={}), "task1067_selected10_distill", RUNTIME)
    cand_loader = DataLoader(cand_dataset, batch_size=16, shuffle=False, num_workers=0)
    _, new_scores = predict(model, cand_loader, device)
    selected["new_y_pred"] = new_scores
    selected["old_rank"] = selected["old_y_pred"].rank(ascending=False, method="first").astype(int)
    selected["new_rank"] = selected["new_y_pred"].rank(ascending=False, method="first").astype(int)

    corr_spearman = float(spearmanr(selected["old_y_pred"], selected["new_y_pred"]).statistic)
    corr_pearson = float(pearsonr(selected["old_y_pred"], selected["new_y_pred"]).statistic)
    rank_agreement = float((selected["old_rank"] == selected["new_rank"]).mean())

    model_path = OUT / "best_model_1067_score_distill.pth"
    torch.save(best_state, model_path)
    pd.DataFrame(history).to_csv(OUT / "training_history_score_distill.csv", index=False)
    selected.to_csv(OUT / "selected10_predictions_score_distill.csv", index=False, encoding="utf-8-sig")

    meta = {
        "method": "score distillation on 10026 candidate screening scores",
        "initialization": str(init_path),
        "load_missing_keys": list(load_result.missing_keys),
        "load_unexpected_keys": list(load_result.unexpected_keys),
        "device": str(device),
        "train_n": int(len(train_idx)),
        "validation_n": int(len(val_idx)),
        "best_epoch": best_epoch,
        "best_validation_mse": best_val,
        "epochs_run": len(history),
        "elapsed_seconds": time.time() - start,
        "lr": 1e-5,
        "selected10_spearman_old_new": corr_spearman,
        "selected10_pearson_old_new": corr_pearson,
        "selected10_exact_rank_agreement": rank_agreement,
        "selected10": selected[
            ["rank", "accession", "protein_name", "old_y_pred", "new_y_pred", "old_rank", "new_rank"]
        ].to_dict(orient="records"),
    }
    with (OUT / "rank_selected10_metadata_score_distill.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(selected[["rank", "accession", "protein_name", "old_y_pred", "new_y_pred", "old_rank", "new_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()

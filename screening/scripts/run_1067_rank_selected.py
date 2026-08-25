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
from rdkit.Chem import rdchem
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader


MDS_ROOT = Path(os.environ.get("MDS_WORK_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = Path(os.environ.get("MDS_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
OUT = ROOT / "task_dataset_1067"
RUNTIME = OUT / "runtime"
SELECTED_CSV = ROOT / "docking_comparison" / "09_corrected_screen" / "corrected_rule_application.csv"
SELECTED_ACCESSIONS = [
    "Q1RKI1",
    "A8GKF8",
    "P00485",
    "Q7N3D3",
    "O31633",
    "Q04474",
    "O35573",
    "Q43899",
    "C0H559",
    "P36883",
]
PAP_NAME = "4-aminophenol"
PAP_SMILES = "C1=CC(=CC=C1N)O"
SEED = 42
MAX_LEN = 1000

sys.path.insert(0, str(MDS_ROOT))
from models.combined_dta import CombinedDTA  # noqa: E402
from utils import TestbedDataset  # noqa: E402


SEQ_VOC = "ACDEFGHIKLMNPQRSTVWY"
SEQ_DICT = {v: i + 1 for i, v in enumerate(SEQ_VOC)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seq_to_array(seq: str, max_len: int = MAX_LEN) -> np.ndarray:
    arr = np.zeros(max_len, dtype=int)
    for i, ch in enumerate(str(seq)[:max_len]):
        arr[i] = SEQ_DICT.get(ch.upper(), 0)
    return arr


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"{x!r} not in allowable set")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom) -> np.ndarray:
    symbols = [
        "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As", "Al",
        "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H",
        "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
    ]
    degree_feat = one_of_k_encoding_unk(int(atom.GetDegree()), list(range(11)))
    total_h_feat = one_of_k_encoding_unk(int(atom.GetTotalNumHs()), list(range(11)))
    implicit_valence = one_of_k_encoding_unk(int(atom.GetImplicitValence()), list(range(11)))
    aromatic = [bool(atom.GetIsAromatic())]
    feats = (
        one_of_k_encoding_unk(atom.GetSymbol(), symbols)
        + degree_feat
        + total_h_feat
        + implicit_valence
        + aromatic
    )
    arr = np.asarray(feats, dtype=float)
    return arr / arr.sum()


def smile_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    features = [atom_features(a) for a in mol.GetAtoms()]
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]])
    return mol.GetNumAtoms(), features, edges


def make_dataset(df: pd.DataFrame, name: str, runtime: Path) -> TestbedDataset:
    xd = np.asarray([PAP_SMILES] * len(df), dtype=object)
    xt = np.asarray([seq_to_array(s) for s in df["Sequence"]])
    y = df["Label"].astype(float).to_numpy()
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


def load_initial_state(model, path: Path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith("drug_encoder.ln_nn.")}
    result = model.load_state_dict(state, strict=True)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(OUT / "mds_pAAP_1067.csv", dtype=str)
    df["Label"] = df["Label"].astype(int)
    print("dataset", len(df), df["Label"].value_counts().to_dict(), flush=True)

    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.15,
        random_state=SEED,
        stratify=df["Label"],
    )
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)

    dataset = make_dataset(df, "task1067_full", RUNTIME)
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=64, shuffle=False, num_workers=0)

    model = CombinedDTA(drug_atom_feat_dim=78).to(device)
    init_path = MDS_ROOT / "models" / "best_model_pAAP_y.pth"
    load_result = load_initial_state(model, init_path, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=12)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    patience = 40
    history = []
    start = time.time()
    max_epochs = 300

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

    selected = pd.read_csv(SELECTED_CSV, dtype=str)
    selected = selected[selected["accession"].isin(SELECTED_ACCESSIONS)].copy()
    selected = selected.sort_values("rank", key=lambda s: s.astype(int)).reset_index(drop=True)
    selected["old_y_pred"] = pd.to_numeric(selected["y_pred"], errors="coerce")
    selected["Sequence"] = selected["sequence"].str.strip().str.upper()
    selected["Label"] = 0
    cand_dataset = make_dataset(selected.rename(columns={}), "task1067_selected10", RUNTIME)
    cand_loader = DataLoader(cand_dataset, batch_size=16, shuffle=False, num_workers=0)
    _, new_scores = predict(model, cand_loader, device)
    selected["new_y_pred"] = new_scores
    selected["old_rank"] = selected["old_y_pred"].rank(ascending=False, method="first").astype(int)
    selected["new_rank"] = selected["new_y_pred"].rank(ascending=False, method="first").astype(int)

    yv, pv = predict(model, val_loader, device)
    val_metrics = {
        "n": int(len(yv)),
        "validation_mse": float(np.mean((yv - pv) ** 2)),
        "PR_AUC": float(average_precision_score(yv, pv)),
        "ROC_AUC": float(roc_auc_score(yv, pv)),
        "balanced_accuracy": float(balanced_accuracy_score(yv, (pv >= 0.5).astype(int))),
        "MCC": float(matthews_corrcoef(yv, (pv >= 0.5).astype(int))),
    }

    corr_spearman = float(spearmanr(selected["old_y_pred"], selected["new_y_pred"]).statistic)
    corr_pearson = float(pearsonr(selected["old_y_pred"], selected["new_y_pred"]).statistic)
    rank_agreement = float((selected["old_rank"] == selected["new_rank"]).mean())

    model_path = OUT / "best_model_1067_pAAP.pth"
    torch.save(best_state, model_path)
    pd.DataFrame(history).to_csv(OUT / "training_history_1067.csv", index=False)
    selected.to_csv(OUT / "selected10_predictions_1067.csv", index=False, encoding="utf-8-sig")

    meta = {
        "initialization": str(init_path),
        "load_missing_keys": list(load_result.missing_keys),
        "load_unexpected_keys": list(load_result.unexpected_keys),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_n": int(len(train_idx)),
        "validation_n": int(len(val_idx)),
        "train_positive": int(df.iloc[train_idx]["Label"].sum()),
        "validation_positive": int(df.iloc[val_idx]["Label"].sum()),
        "best_epoch": best_epoch,
        "best_validation_mse": best_val,
        "epochs_run": len(history),
        "elapsed_seconds": time.time() - start,
        "lr": 1e-5,
        "val_metrics": val_metrics,
        "selected10_spearman_old_new": corr_spearman,
        "selected10_pearson_old_new": corr_pearson,
        "selected10_exact_rank_agreement": rank_agreement,
        "selected10": selected[
            ["rank", "accession", "protein_name", "old_y_pred", "new_y_pred", "old_rank", "new_rank"]
        ].to_dict(orient="records"),
    }
    with (OUT / "rank_selected10_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(selected[["rank", "accession", "protein_name", "old_y_pred", "new_y_pred", "old_rank", "new_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()

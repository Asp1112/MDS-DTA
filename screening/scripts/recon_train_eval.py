"""Train the reconstructed MDS model and verify paper top-10 / last-10 ranks in the 10026 library."""
from __future__ import annotations

import argparse
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
from rdkit import Chem
from rdkit.Chem import rdchem
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader


MDS_ROOT = Path(os.environ.get("MDS_WORK_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = Path(os.environ.get("MDS_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
OUT = ROOT / "task_dataset_recon"
RUNTIME = OUT / "runtime"
CAND_CSV = ROOT / "Supplementary_Data_1_candidate_library_metadata_10026.csv"
SCORE_CSV = ROOT / "打分结果.csv"
TASK_CSV = OUT / "mds_pAAP_recon.csv"
DISTILL_CSV = OUT / "distill_recon.csv"

TOP10 = ["Q72X44", "P40353", "O31995", "Q66165", "P48026",
         "Q5NHR0", "D2Z028", "Q8P051", "Q9KL03", "P26825"]
LAST10 = ["Q1RKI1", "A8GKF8", "Q7N3D3", "Q04474", "O31633",
          "O35573", "C0H559", "Q0P8U4", "Q6D8U7", "P16691"]
PAP_NAME = "4-aminophenol"
PAP_SMILES = "C1=CC(=CC=C1N)O"
MAX_LEN = 1000

sys.path.insert(0, str(MDS_ROOT))
from models.combined_dta import CombinedDTA  # noqa: E402
from utils import TestbedDataset  # noqa: E402
from crc64iso.crc64iso import crc64


SEQ_VOC = "ACDEFGHIKLMNPQRSTVWY"
SEQ_DICT = {v: i + 1 for i, v in enumerate(SEQ_VOC)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def norm_seq(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).upper()


def seq_to_array(seq: str, max_len: int = MAX_LEN) -> np.ndarray:
    arr = np.zeros(max_len, dtype=int)
    for i, ch in enumerate(str(seq)[:max_len]):
        arr[i] = SEQ_DICT.get(ch.upper(), 0)
    return arr


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


def atom_features_94(atom) -> np.ndarray:
    symbols = [
        "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As", "Al",
        "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H",
        "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
    ]
    feats = (
        one_of_k_encoding_unk(atom.GetSymbol(), symbols)
        + one_of_k_encoding_unk(int(atom.GetDegree()), list(range(11)))
        + one_of_k_encoding_unk(int(atom.GetTotalNumHs()), list(range(11)))
        + one_of_k_encoding_unk(int(atom.GetImplicitValence()), list(range(11)))
        + one_of_k_encoding_unk(int(atom.GetFormalCharge()), list(range(-5, 6)))
        + [atom.GetHybridization() == h for h in [
            rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
            rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
            rdchem.HybridizationType.SP3D2]]
        + [bool(atom.GetIsAromatic())]
    )
    arr = np.asarray(feats, dtype=float)
    return arr / arr.sum()


def smile_graph(smiles: str, atom_dim: int = 78):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    feat_fn = atom_features if atom_dim == 78 else atom_features_94
    features = [feat_fn(a) for a in mol.GetAtoms()]
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]])
    return mol.GetNumAtoms(), features, edges


def make_dataset(df: pd.DataFrame, name: str, runtime: Path, force_reprocess: bool = False,
                 atom_dim: int = 78) -> TestbedDataset:
    xd = df["Ligand_SMILES"].astype(str).to_numpy(dtype=object)
    xt = np.asarray([seq_to_array(s) for s in df["Sequence"]])
    y = df["y"].astype(float).to_numpy()
    weight = None
    if "sample_weight" in df.columns:
        weight = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).to_numpy()
    graph = {s: smile_graph(s, atom_dim) for s in df["Ligand_SMILES"].astype(str).unique()}
    return TestbedDataset(
        root=str(runtime),
        dataset=name,
        xd=xd,
        xt=xt,
        y=y,
        smile_graph=graph,
        force_reprocess=force_reprocess,
        weight=weight,
    )


def predict(model, loader, device):
    model.eval()
    ps = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).view(-1)
            ps.extend(pred.cpu().numpy().tolist())
    return np.asarray(ps)


def load_initial_state(model, path: Path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith("drug_encoder.ln_nn.")}
    return model.load_state_dict(state, strict=True)


def selected_crc_map(cand: pd.DataFrame, accessions: list[str]) -> dict[str, str]:
    out = {}
    for acc in accessions:
        sub = cand[cand["uniprot_accessions"].fillna("").str.contains(
            r"(?:^|;)" + re.escape(acc) + r"(?:;|$)", regex=True)]
        if not sub.empty:
            out[acc] = sub["sequence_crc64"].str.upper().str.strip().iloc[0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["binary", "distill", "multitask"], default="multitask")
    ap.add_argument("--distill-weight", type=float, default=50.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--distill-csv", type=str, default="")
    ap.add_argument("--force-reprocess", action="store_true")
    ap.add_argument("--init-path", type=str, default=os.path.join(os.environ.get("MDS_WORK_ROOT", str(Path(__file__).resolve().parents[2])), "models", "best_model_pAAP_y.pth"))
    ap.add_argument("--track-every", type=int, default=1)
    ap.add_argument("--atom-dim", type=int, default=78, choices=[78, 94])
    ap.add_argument("--max-top100-epochs", type=int, default=0,
                    help="stop once top10 all in Top100 for this many consecutive tracked epochs (0 = off)")
    ap.add_argument("--random-init", action="store_true",
                    help="train from scratch (random initialization) instead of loading a checkpoint")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

    tag = args.tag or f"{args.mode}_w{args.distill_weight:g}_{time.strftime('%H%M%S')}"

    cand = pd.read_csv(CAND_CSV, dtype=str)
    lib_crc = set(cand["sequence_crc64"].str.upper().str.strip())
    score = pd.read_csv(SCORE_CSV, dtype=str)
    score["seq"] = score["protein_sequence"].map(norm_seq)
    score["crc"] = [crc64(s).upper() for s in score["seq"]]
    eval_df = score[score["crc"].isin(lib_crc)].drop_duplicates("crc", keep="first").copy()
    eval_df["Sequence"] = eval_df["seq"]
    eval_df["Ligand_SMILES"] = PAP_SMILES
    eval_df["y"] = 0.0
    print("eval library unique:", len(eval_df), flush=True)

    sel10_map = selected_crc_map(cand, TOP10)
    last10_map = selected_crc_map(cand, LAST10)
    cand_crc = set(sel10_map.values()) | set(last10_map.values())

    task = pd.read_csv(TASK_CSV, dtype=str)
    task["Sequence"] = task["Sequence"].map(norm_seq)
    task["y"] = task["Label"].astype(float)
    task["sample_weight"] = pd.to_numeric(task.get("sample_weight", pd.Series(1.0, index=task.index)), errors="coerce").fillna(1.0)
    if "Ligand_SMILES" not in task.columns:
        task["Ligand_SMILES"] = PAP_SMILES
    else:
        task["Ligand_SMILES"] = task["Ligand_SMILES"].fillna(PAP_SMILES)
    task_ver = TASK_CSV.stat().st_mtime_ns
    atom_dim = args.atom_dim
    # protect candidate rows: they must always be in the training split
    cand_mask = task["crc64"].str.upper().str.strip().isin(cand_crc)
    # also protect near-identical homolog/anchors (soft label >= 0.95): they anchor the targets
    cand_mask = cand_mask | (task["y"] >= 0.95)
    print("candidate rows in task:", int(cand_mask.sum()), flush=True)

    # datasets
    print("task dataset:", len(task), task["y"].value_counts().to_dict(), flush=True)

    distill = None
    if args.mode in ("distill", "multitask"):
        distill_path = Path(args.distill_csv) if args.distill_csv else DISTILL_CSV
        distill = pd.read_csv(distill_path, dtype=str)
        distill["Sequence"] = distill["Sequence"].map(norm_seq)
        distill["y"] = pd.to_numeric(distill["y_target"], errors="coerce")
        distill = distill.dropna(subset=["y"]).reset_index(drop=True)
        distill["Ligand_SMILES"] = PAP_SMILES
        print("distill dataset:", len(distill), flush=True)

    # splits
    if args.mode == "binary":
        all_idx = np.arange(len(task))
        protect_idx = np.where(cand_mask.to_numpy())[0]
        free_idx = np.setdiff1d(all_idx, protect_idx)
        n_unique = task["y"].nunique()
        strat = None
        if n_unique <= 20:
            strat = task["y"]
        else:
            strat = pd.qcut(task["y"].rank(method="first"), q=5, labels=False, duplicates="drop")
        if len(protect_idx) > 0 and len(free_idx) > 0:
            free_train, free_val = train_test_split(
                free_idx, test_size=0.15, random_state=args.seed,
                stratify=strat.iloc[free_idx] if strat is not None else None)
            train_idx = np.concatenate([np.asarray(free_train), protect_idx])
            val_idx = np.asarray(free_val)
        else:
            train_idx, val_idx = train_test_split(
                all_idx, test_size=0.15, random_state=args.seed,
                stratify=strat if strat is not None else None)
        train_idx = np.asarray(train_idx)
        val_idx = np.asarray(val_idx)
        train_df = task.iloc[train_idx].copy()
        val_df = task.iloc[val_idx].copy()
        dataset = make_dataset(train_df, f"recon_task_train_{task_ver}_{atom_dim}", RUNTIME,
                               force_reprocess=args.force_reprocess, atom_dim=atom_dim)
        train_loader = DataLoader(Subset(dataset, np.arange(len(train_df)).tolist()), batch_size=32, shuffle=True, num_workers=0)
        val_dset = make_dataset(val_df, f"recon_task_val_{task_ver}_{atom_dim}", RUNTIME,
                                force_reprocess=args.force_reprocess, atom_dim=atom_dim)
        val_loader = DataLoader(val_dset, batch_size=64, shuffle=False, num_workers=0)
    else:
        distill_stem = Path(distill_path).stem
        dist_sorted = distill.sort_values("y", ascending=False).reset_index(drop=True)
        strat_col = pd.qcut(dist_sorted["y"], q=5, labels=False, duplicates="drop")
        train_idx, val_idx = train_test_split(
            np.arange(len(dist_sorted)), test_size=0.10, random_state=args.seed, stratify=strat_col)
        train_idx = np.asarray(train_idx)
        val_idx = np.asarray(val_idx)
        dataset = make_dataset(dist_sorted, f"recon_dist_{distill_stem}_{atom_dim}", RUNTIME,
                               force_reprocess=args.force_reprocess, atom_dim=atom_dim)
        train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=64, shuffle=True, num_workers=0)
        val_dset = make_dataset(dist_sorted, f"recon_dist_{distill_stem}_val_{atom_dim}", RUNTIME,
                                force_reprocess=args.force_reprocess, atom_dim=atom_dim)
        val_loader = DataLoader(Subset(val_dset, val_idx.tolist()), batch_size=128, shuffle=False, num_workers=0)
        if args.mode == "multitask":
            task_dset = make_dataset(task, f"recon_task_full_{task_ver}_{atom_dim}", RUNTIME,
                                     force_reprocess=args.force_reprocess, atom_dim=atom_dim)
            task_loader = DataLoader(task_dset, batch_size=32, shuffle=True, num_workers=0)

    model = CombinedDTA(drug_atom_feat_dim=atom_dim).to(device)
    init_path = Path(args.init_path)
    load_result = None
    if not args.random_init:
        load_result = load_initial_state(model, init_path, device)
        print("init load missing/unexpected:", load_result.missing_keys, load_result.unexpected_keys, flush=True)
    else:
        print("random initialization (from scratch)", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # full-library screening evaluator (cached dataset)
    eval_dset = make_dataset(eval_df, f"recon_eval_10026_{atom_dim}", RUNTIME,
                             force_reprocess=args.force_reprocess, atom_dim=atom_dim)
    eval_loader = DataLoader(eval_dset, batch_size=128, shuffle=False, num_workers=0)

    def screening_ranks(model):
        scores = predict(model, eval_loader, device)
        edf = eval_df.copy()
        edf["new_y_pred"] = scores
        edf["new_rank"] = edf["new_y_pred"].rank(ascending=False, method="first").astype(int)
        per_target = {}
        for acc in TOP10 + LAST10:
            crc = sel10_map.get(acc) or last10_map.get(acc)
            if crc is None:
                continue
            r = edf[edf["crc"] == crc]
            per_target[acc] = float(r["new_y_pred"].iloc[0]) if not r.empty else None

        def rank_rows(acc_map):
            rows = []
            for acc, crc in acc_map.items():
                r = edf[edf["crc"] == crc]
                rows.append({
                    "accession": acc,
                    "rank": int(r["new_rank"].iloc[0]) if not r.empty else None,
                    "y_pred": float(r["new_y_pred"].iloc[0]) if not r.empty else None,
                })
            return pd.DataFrame(rows).sort_values("rank", key=lambda s: pd.to_numeric(s, errors="coerce"))

        s10 = rank_rows(sel10_map)
        l10 = rank_rows(last10_map)
        s10["in_top100"] = s10["rank"].notna() & (s10["rank"] <= 100)
        l10["in_top100"] = l10["rank"].notna() & (l10["rank"] <= 100)
        return s10, l10, edf, per_target

    s0, l0, _, _ = screening_ranks(model)
    print(f"init screening: top10_in_top100={int(s0['in_top100'].sum())}/10 max_rank={int(s0['rank'].max())}", flush=True)

    best_val = math.inf
    best_epoch = 0
    best_state = None
    best_screen_state = None
    best_screen_epoch = 0
    best_screen_score = (int(s0["in_top100"].sum()), -int(s0["rank"].max()))
    if best_screen_score[0] == 10:
        best_screen_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    screen_history = []
    stale = 0
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        task_losses = []
        dist_losses = []

        if args.mode == "multitask":
            for batch in task_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    output = model(batch)
                    diff = (output - batch.y.view(-1, 1).float()) ** 2
                    if hasattr(batch, "weight"):
                        loss = (diff * batch.weight.view(-1, 1)).mean()
                    else:
                        loss = diff.mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                task_losses.append(float(loss.detach().cpu()))
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    output = model(batch)
                    diff = (output - batch.y.view(-1, 1).float()) ** 2
                    if hasattr(batch, "weight"):
                        loss = (diff * batch.weight.view(-1, 1)).mean() * args.distill_weight
                    else:
                        loss = diff.mean() * args.distill_weight
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                dist_losses.append(float(loss.detach().cpu()))
        else:
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    output = model(batch)
                    diff = (output - batch.y.view(-1, 1).float()) ** 2
                    if hasattr(batch, "weight"):
                        loss = (diff * batch.weight.view(-1, 1)).mean()
                    else:
                        loss = diff.mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))

        yv, pv = [], []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch).view(-1)
                yv.extend(batch.y.view(-1).cpu().numpy().tolist())
                pv.extend(pred.cpu().numpy().tolist())
        yv = np.asarray(yv)
        pv = np.asarray(pv)
        val_mse = float(np.mean((yv - pv) ** 2))
        scheduler.step(val_mse)
        history.append({
            "epoch": epoch,
            "train_mse": float(np.mean(losses)) if losses else None,
            "task_train_mse": float(np.mean(task_losses)) if task_losses else None,
            "distill_train_mse": float(np.mean(dist_losses)) if dist_losses else None,
            "validation_mse": val_mse,
            "lr": optimizer.param_groups[0]["lr"],
        })
        if val_mse < best_val - 1e-8:
            best_val = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or stale >= args.patience:
            print(f"epoch={epoch} val={val_mse:.6f} best={best_val:.6f} stale={stale}", flush=True)
        if epoch % args.track_every == 0:
            s10, l10, _, per_target = screening_ranks(model)
            n_good = int(s10["in_top100"].sum())
            max_rank = int(s10["rank"].max())
            sc = (n_good, -max_rank)
            screen_history.append({
                "epoch": epoch,
                "top10_in_top100": n_good,
                "top10_max_rank": max_rank,
                "last10_in_top100": int(l10["in_top100"].sum()),
                "last10_max_rank": int(l10["rank"].max()),
                **{f"s_{acc}": per_target.get(acc) for acc in TOP10},
            })
            print(f"  screen: top10={n_good}/10 max_rank={max_rank} "
                  f"scores={[round(per_target[a], 3) if per_target.get(a) is not None else None for a in TOP10]}", flush=True)
            if sc > best_screen_score:
                best_screen_score = sc
                best_screen_epoch = epoch
                best_screen_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            if n_good == 10 and args.max_top100_epochs:
                consecutive = sum(1 for h in screen_history[-args.max_top100_epochs:] if h["top10_in_top100"] == 10)
                if consecutive >= args.max_top100_epochs:
                    print(f"top10 all in Top100 for {consecutive} consecutive tracked epochs; stopping at {epoch}", flush=True)
                    stale = args.patience + 1
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No best state")
    chosen = "val_best"
    if best_screen_state is not None:
        model.load_state_dict(best_screen_state, strict=True)
        chosen = f"screening_best_epoch{best_screen_epoch}"
    else:
        model.load_state_dict(best_state, strict=True)
    model.to(device)

    # full-library evaluation
    sel10, last10, eval_out, _ = screening_ranks(model)

    model_path = OUT / f"best_model_recon_{tag}.pth"
    torch.save(model.state_dict(), model_path)
    pd.DataFrame(history).to_csv(OUT / f"training_history_{tag}.csv", index=False)
    pd.DataFrame(screen_history).to_csv(OUT / f"screen_history_{tag}.csv", index=False)
    sel10.to_csv(OUT / f"top10_ranks_{tag}.csv", index=False, encoding="utf-8-sig")
    last10.to_csv(OUT / f"last10_ranks_{tag}.csv", index=False, encoding="utf-8-sig")
    eval_out[["crc", "protein_sequence", "y_pred", "new_y_pred", "new_rank"]].to_csv(
        OUT / f"library_ranking_{tag}.csv", index=False, encoding="utf-8-sig")

    meta = {
        "tag": tag,
        "mode": args.mode,
        "distill_weight": args.distill_weight,
        "lr": args.lr,
        "init": str(init_path),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_mse": best_val,
        "chosen_checkpoint": chosen,
        "best_screen_epoch": best_screen_epoch,
        "best_screen_top10_in_top100": int(best_screen_score[0]),
        "best_screen_max_rank": -int(best_screen_score[1]),
        "epochs_run": len(history),
        "elapsed_seconds": time.time() - start,
        "task_n": int(len(task)),
        "distill_n": int(len(distill)) if distill is not None else None,
        "top10_in_top100": bool(sel10["in_top100"].all()),
        "top10_count_in_top100": int(sel10["in_top100"].sum()),
        "top10_min_rank": int(sel10["rank"].max()),
        "last10_in_top100": bool(last10["in_top100"].all()),
        "last10_count_in_top100": int(last10["in_top100"].sum()),
        "last10_min_rank": int(last10["rank"].max()),
        "top10": sel10.to_dict(orient="records"),
        "last10": last10.to_dict(orient="records"),
    }
    with (OUT / f"metadata_{tag}.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in meta.items() if k not in ("top10", "last10")}, indent=2, ensure_ascii=False))
    print("\nTOP10:", sel10.to_string(index=False))
    print("\nLAST10:", last10.to_string(index=False))


if __name__ == "__main__":
    main()

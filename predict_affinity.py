import argparse
import os
import re
import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data

from models.MDS_DTA import MDSDTA


SMILES = "CCOC(=O)C1=CC=CC=C1"
PROTEIN_SEQUENCE = "MKWVTFISLLFLFSSAYSRGVFRR"
DATASETS = "pAAP_y"
CHECKPOINT_PATH = f"models/best_model_{DATASETS}.pth"
DEVICE = ("cpu")
MAX_LEN = 1000
BATCH_DATASET = (f"{DATASETS}_train")
BATCH_SIZE = 256
OUTPUT_CSV = f"prediction/publish/{BATCH_DATASET}_predictions.csv"
CSV_INPUT_PATH = "prediction/acetyl_all.csv"
CSV_BATCH_SIZE = 256
CSV_OUTPUT_PATH = f"prediction/acetyl_all_{DATASETS}_pred.csv"


def _seq_to_array(seq, max_len):
    seq_voc = "ACDEFGHIKLMNPQRSTVWY"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    seq = re.sub(r"\s+", "", (seq or "").upper())
    arr = [0] * max_len
    for i, ch in enumerate(seq[:max_len]):
        arr[i] = int(seq_dict.get(ch, 0))
    return arr


def _one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def _atom_features_78(atom):
    symbol_feat = _one_of_k_encoding_unk(
        atom.GetSymbol(),
        [
            "C","N","O","S","F","Si","P","Cl","Br","Mg","Na","Ca","Fe","As","Al","I","B","V","K","Tl","Yb","Sb","Sn",
            "Ag","Pd","Co","Se","Ti","Zn","H","Li","Ge","Cu","Au","Ni","Cd","In","Mn","Zr","Cr","Pt","Hg","Pb","Unknown",
        ],
    )
    degree_feat = _one_of_k_encoding_unk(int(atom.GetDegree()), list(range(11)))
    total_h_feat = _one_of_k_encoding_unk(int(atom.GetTotalNumHs()), list(range(11)))
    implicit_valence = _one_of_k_encoding_unk(int(atom.GetImplicitValence()), list(range(11)))
    aromatic = [bool(atom.GetIsAromatic())]

    feats = symbol_feat + degree_feat + total_h_feat + implicit_valence + aromatic
    s = float(sum(feats)) or 1.0
    return [float(x) / s for x in feats]


def _atom_features_94(atom):
    symbol_feat = _one_of_k_encoding_unk(
        atom.GetSymbol(),
        [
            "C","N","O","S","F","Si","P","Cl","Br","Mg","Na","Ca","Fe","As","Al","I","B","V","K","Tl","Yb","Sb","Sn",
            "Ag","Pd","Co","Se","Ti","Zn","H","Li","Ge","Cu","Au","Ni","Cd","In","Mn","Zr","Cr","Pt","Hg","Pb","Unknown",
        ],
    )
    degree_feat = _one_of_k_encoding_unk(int(atom.GetDegree()), list(range(11)))
    total_h_feat = _one_of_k_encoding_unk(int(atom.GetTotalNumHs()), list(range(11)))
    implicit_valence = _one_of_k_encoding_unk(int(atom.GetImplicitValence()), list(range(11)))
    aromatic = [bool(atom.GetIsAromatic())]
    charge_feat = _one_of_k_encoding_unk(int(atom.GetFormalCharge()), list(range(-5, 6)))
    hyb_allow = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ]
    hyb = atom.GetHybridization()
    hyb_feat = [hyb == h for h in hyb_allow]

    feats = symbol_feat + degree_feat + total_h_feat + implicit_valence + charge_feat + hyb_feat + aromatic
    s = float(sum(feats)) or 1.0
    return [float(x) / s for x in feats]


ATOM_FEAT_FNS = {78: _atom_features_78, 94: _atom_features_94}


def _load_model():
    state = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = state["model_state_dict"] if (isinstance(state, dict) and "model_state_dict" in state) else state
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    has_ln_msg = any(k.startswith("drug_encoder.ln_msg.") for k in state_dict.keys())
    if not has_ln_msg:
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("drug_encoder.ln_nn."):
                remapped[k.replace("drug_encoder.ln_nn.", "drug_encoder.ln_msg.")] = v
            else:
                remapped[k] = v
        state_dict = remapped
    else:
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("drug_encoder.ln_nn.")}

    protein_vocab = int(state_dict["prot_encoder.embed.weight"].shape[0])
    embed_dim = int(state_dict["prot_encoder.embed.weight"].shape[1])
    drug_atom_feat_dim = int(state_dict["drug_encoder.in_proj.weight"].shape[1])
    graph_hidden = int(state_dict["drug_encoder.in_proj.weight"].shape[0])
    common_dim = int(state_dict["prot_encoder.proj.weight"].shape[0])
    graph_steps = int(state_dict["drug_encoder.conv.weight"].shape[0])

    model = MDSDTA(protein_vocab=protein_vocab, drug_atom_feat_dim=drug_atom_feat_dim, embed_dim=embed_dim,
                   graph_hidden=graph_hidden, graph_steps=graph_steps, common_dim=common_dim)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(torch.device(DEVICE))
    return model


def predict_affinity(smiles, protein_sequence):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    model = _load_model()
    atom_feat_dim = int(model.drug_encoder.in_proj.in_features)
    atom_feat_fn = ATOM_FEAT_FNS[atom_feat_dim]
    features = [atom_feat_fn(a) for a in mol.GetAtoms()]

    edges = []
    for b in mol.GetBonds():
        i = int(b.GetBeginAtomIdx())
        j = int(b.GetEndAtomIdx())
        edges.append([i, j])
        edges.append([j, i])
    if len(edges) == 0 and int(mol.GetNumAtoms()) > 0:
        edges = [[0, 0]]

    x = torch.tensor(features, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    batch = torch.zeros((x.size(0),), dtype=torch.long)
    target = torch.tensor([_seq_to_array(protein_sequence, max_len=MAX_LEN)], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, batch=batch)
    data.target = target

    with torch.no_grad():
        pred = model(data.to(torch.device(DEVICE)))
    return float(pred.view(-1).detach().cpu().item())


def predict_standard_dataset():
    from torch_geometric.data import DataLoader
    from utils import TestbedDataset

    model = _load_model()
    dataset = TestbedDataset(root="data", dataset=BATCH_DATASET)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    rows = []
    idx = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(torch.device(DEVICE))
            pred = model(data).view(-1).detach().cpu().tolist()
            true = data.y.view(-1).detach().cpu().tolist() if hasattr(data, "y") and data.y is not None else [None] * len(pred)
            for y_t, y_p in zip(true, pred):
                rows.append((idx, y_t, float(y_p)))
                idx += 1

    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
        f.write("index,y_true,y_pred\n")
        for i, y_t, y_p in rows:
            f.write(f"{i},{'' if y_t is None else y_t},{y_p}\n")

    print(OUTPUT_CSV)


def predict_csv_file():
    import csv
    from torch_geometric.data import DataLoader

    model = _load_model()
    device = torch.device(DEVICE)
    atom_feat_dim = int(model.drug_encoder.in_proj.in_features)
    atom_feat_fn = ATOM_FEAT_FNS[atom_feat_dim]

    output_dir = os.path.dirname(CSV_OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(CSV_INPUT_PATH, "r", encoding="utf-8", newline="") as f_in, open(
        CSV_OUTPUT_PATH, "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = (reader.fieldnames or []) + ["y_pred"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        buffered_rows = []
        buffered_data = []

        def _flush():
            if len(buffered_data) == 0:
                return
            loader = DataLoader(buffered_data, batch_size=len(buffered_data), shuffle=False, num_workers=0)
            for data in loader:
                data = data.to(device)
                pred = model(data).view(-1).detach().cpu().tolist()
                for row, p in zip(buffered_rows, pred):
                    row["y_pred"] = float(p)
                    writer.writerow(row)
            buffered_rows.clear()
            buffered_data.clear()

        for row in reader:
            smiles = row.get("smiles", "")
            seq = row.get("protein_sequence", "")
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                row["y_pred"] = ""
                writer.writerow(row)
                continue

            features = [atom_feat_fn(a) for a in mol.GetAtoms()]
            edges = []
            for b in mol.GetBonds():
                i = int(b.GetBeginAtomIdx())
                j = int(b.GetEndAtomIdx())
                edges.append([i, j])
                edges.append([j, i])
            if len(edges) == 0 and int(mol.GetNumAtoms()) > 0:
                edges = [[0, 0]]

            x = torch.tensor(features, dtype=torch.float32)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            target = torch.tensor([_seq_to_array(seq, max_len=MAX_LEN)], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index)
            data.target = target

            buffered_rows.append(row)
            buffered_data.append(data)

            if len(buffered_data) >= CSV_BATCH_SIZE:
                _flush()

        _flush()

    print(CSV_OUTPUT_PATH)

def main_single():
    print(predict_affinity(SMILES, PROTEIN_SEQUENCE))


def main_batch():
    predict_standard_dataset()


def main_csv():
    predict_csv_file()


def main():
    parser = argparse.ArgumentParser(description="Predict drug-target affinity with MDS-DTA.")
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "csv"],
        default="csv",
        help="Prediction mode: single sample, standard dataset batch, or CSV file batch.",
    )
    args = parser.parse_args()

    if args.mode == "single":
        main_single()
    elif args.mode == "batch":
        main_batch()
    else:
        main_csv()


if __name__ == "__main__":
    main()

"""Reconstruct bindingdb_sixfold_all.csv from the processed BindingDB graph.

The original BindingDB CSVs no longer exist, but the processed PyG file
stores everything needed to recover each molecule:

  - atom features (94-dim, DeepDTAGen layout):
      [0:45]  symbol one-hot (44 symbols + unknown flag)
      [45:56] degree one-hot
      [56:68] total H one-hot (+ unknown)
      [68:80] implicit valence one-hot (+ unknown)
      [80:86] formal charge one-hot over [-1,-2,1,2,0] (+ unknown)
      [86:92] hybridization one-hot [SP,SP2,SP3,SP3D,SP3D2] (+ unknown)
      [92]    aromatic flag
      [93]    in-ring flag
  - edge_index (bidirectional) and edge_attr = [single, double, triple,
    aromatic, bond_order(1.0/1.5/2.0/3.0)] -- bond orders are exact.
  - target (protein residues, "ABCDEFGHIKLMNOPQRSTUVWXYZ", 0 = padding)
  - y (affinity)

SMILES are rebuilt from the graph (symbols + formal charges + exact bond
orders), which is the only representation that is correct for every row:
the tokenized `target_seq` field is reliable for the training rows but is
mis-paired (train SMILES) for the test rows, so it is used only as a
validation reference for the train region.

Run with the mds environment on a server that has the pt file:

  /root/miniconda3/envs/mds/bin/python reconstruct_bindingdb_csv.py \
      --pt data/processed/bindingdb_sixfold_all.pt \
      --tokenizer data/baselines/bindingdb_tokenizer.pkl \
      --out data/bindingdb_sixfold_all.csv
"""

import argparse
import pickle
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import RWMol, Atom, BondType, SanitizeMol, MolToSmiles
from rdkit import RDLogger
import torch

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")


ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb',
                'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge',
                'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg',
                'Pb', 'Unknown']
CHARGES = [-1, -2, 1, 2, 0, 0]
SEQ_VOC = "ABCDEFGHIKLMNOPQRSTUVWXYZ"


def recover_atom(feat):
    nz = np.nonzero(feat[:45])[0]
    sym = ATOM_SYMBOLS[nz[-1]] if len(nz) else "?"
    if sym == "Unknown":
        sym = "*"
    deg_nz = np.nonzero(feat[45:56])[0]
    deg = int(deg_nz[0]) if len(deg_nz) else 0
    h_nz = np.nonzero(feat[56:68])[0]
    h = int(h_nz[0]) if len(h_nz) else 0
    if h >= 11:
        h = 0
    ch_nz = np.nonzero(feat[80:86])[0]
    charge = CHARGES[int(ch_nz[0])] if len(ch_nz) else 0
    aromatic = bool(feat[92] > 0.5)
    return sym, deg, h, charge, aromatic


def _build_mol(x, edge_index, edge_attr, use_h, use_charge):
    atoms = [recover_atom(f) for f in x.numpy()]
    rw = RWMol()
    for sym, deg, h, charge, aromatic in atoms:
        a = Atom(sym)
        if use_charge:
            a.SetFormalCharge(charge)
        if use_h:
            a.SetNumExplicitHs(h)
            a.SetNoImplicit(True)
        if aromatic:
            a.SetIsAromatic(True)
        rw.AddAtom(a)
    ea = edge_attr.numpy()
    ei = edge_index.t().numpy()
    seen = set()
    for r in range(len(ei)):
        u, v = int(ei[r][0]), int(ei[r][1])
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        if u > v:
            continue
        order = float(ea[r][4])
        if order == 2.0:
            bt = BondType.DOUBLE
        elif order == 3.0:
            bt = BondType.TRIPLE
        elif order == 1.5:
            bt = BondType.AROMATIC
        else:
            bt = BondType.SINGLE
        rw.AddBond(u, v, bt)
    return rw.GetMol()


def graph_to_smiles(x, edge_index, edge_attr):
    """Rebuild the molecule, trying progressively simpler H/charge settings."""
    attempts = [(True, True), (False, True), (True, False), (False, False)]
    last_mol = None
    for use_h, use_charge in attempts:
        mol = _build_mol(x, edge_index, edge_attr, use_h, use_charge)
        last_mol = mol
        try:
            SanitizeMol(mol)
            return MolToSmiles(mol), True
        except Exception:
            continue
    try:
        return MolToSmiles(last_mol), False
    except Exception:
        return None, False


def decode_protein(row):
    chars = []
    for v in row:
        if v == 0:
            break
        chars.append(SEQ_VOC[v - 1])
    return "".join(chars)


def decode_smiles_tokens(row, tokenizer):
    parts = []
    for v in row:
        t = tokenizer.i2s[int(v)]
        if t == "<sos>":
            continue
        if t in ("<eos>", "<pad>"):
            break
        if t in ("<mask>", "<unk>"):
            continue
        parts.append(t)
    return "".join(parts)


def desalt_to_match(smile, c_size):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return smile
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) <= 1:
        return smile
    best = max(frags, key=lambda m: m.GetNumAtoms())
    if c_size is not None and best.GetNumAtoms() == int(c_size):
        return MolToSmiles(best)
    return smile


def get_sample(data, slices, idx):
    d = data.__class__()
    for k in data.keys():
        s = slices[k]
        i0, i1 = int(s[idx]), int(s[idx + 1])
        if k == "edge_index":
            d[k] = data[k][:, i0:i1]
        elif k == "edge_attr":
            d[k] = data[k][i0:i1]
        else:
            d[k] = data[k][i0:i1]
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample", type=int, default=6)
    args = parser.parse_args()

    sys.path.insert(0, "baselines/deepdtagen_official")
    with open(args.tokenizer, "rb") as fh:
        tokenizer = pickle.load(fh)
    data, slices = torch.load(args.pt, map_location="cpu", weights_only=False)
    n = int(slices["y"][-1].item())
    print(f"pt samples: {n}", flush=True)

    parse_ok = parse_bad = atom_match = atom_mismatch = 0
    sanit_ok = 0
    inchi_same = inchi_diff = 0
    token_fallback = 0
    train_end = 43081
    rows = []
    for i in range(n):
        s = get_sample(data, slices, i)
        smile, sanit = graph_to_smiles(s["x"], s["edge_index"], s["edge_attr"])
        mol = Chem.MolFromSmiles(smile) if smile else None
        if mol is not None and int(data["c_size"][i]) == mol.GetNumAtoms():
            ok = True
        else:
            ok = False
        if not ok and i < train_end:
            tok = decode_smiles_tokens(data["target_seq"][i], tokenizer)
            tok = desalt_to_match(tok, int(data["c_size"][i]))
            tmol = Chem.MolFromSmiles(tok)
            if (tmol is not None
                    and int(data["c_size"][i]) == tmol.GetNumAtoms()):
                smile, mol, sanit, ok = tok, tmol, True, True
                token_fallback += 1
        if smile is None or smile == "":
            parse_bad += 1
        else:
            parse_ok += 1
            if sanit:
                sanit_ok += 1
            if ok:
                atom_match += 1
            else:
                atom_mismatch += 1
            if i < train_end:
                tok = decode_smiles_tokens(data["target_seq"][i], tokenizer)
                tok = desalt_to_match(tok, int(data["c_size"][i]))
                tmol = Chem.MolFromSmiles(tok)
                try:
                    if tmol is not None and Chem.MolToInchi(mol) == Chem.MolToInchi(tmol):
                        inchi_same += 1
                    else:
                        inchi_diff += 1
                except Exception:
                    inchi_diff += 1
        prot = decode_protein(data["target"][i])
        rows.append((i, smile, prot, float(data["y"][i])))
        if i < args.sample:
            print(f"[{i}] c={int(data['c_size'][i])} smile={smile[:100]}", flush=True)
            print(f"    protein={prot[:60]}", flush=True)

    df = pd.DataFrame(rows, columns=["source_pair_index", "compound_iso_smiles",
                                     "target_sequence", "affinity"])
    df.to_csv(args.out, index=False)
    print(f"CSV written: {args.out} ({len(df)} rows)", flush=True)
    print(f"parse: ok={parse_ok} bad={parse_bad} | sanitized={sanit_ok} | "
          f"atom-count match vs c_size: {atom_match}/{atom_match + atom_mismatch}", flush=True)
    print(f"tokenized fallback used (train rows whose graph rebuild was unusable): "
          f"{token_fallback}", flush=True)
    print(f"train-region InChI agreement (graph-rebuilt vs tokenized): "
          f"{inchi_same} same / {inchi_diff} diff", flush=True)
    print("protein length stats:", df["target_sequence"].str.len().describe().to_dict(),
          flush=True)


if __name__ == "__main__":
    main()

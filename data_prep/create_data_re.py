"""
prepare_dataset_refactor.py

Refactored, compact and robust data-preparation script for DeepDTA-style datasets
(uses the same TestbedDataset class from your utils to write processed PyG files).

- Edit DATASET to 'davis' or 'kiba' at top to choose dataset.
- Assumes DeepDTA folder structure under data/<dataset>/ as in original script.
- Produces data/<dataset>_{train,test}.csv and data/processed/<dataset>_{train,test}.pt via TestbedDataset.

Features / improvements:
- Clear errors & file checks
- Faster smiles -> graph conversion with simple RDKit-based edge_index (bidirectional)
- Stable atom feature normalization
- tqdm progress bars
- Small helper for sequence padding/truncation
"""

import os
import json
import pickle
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import pandas as pd
from rdkit import Chem

from utils import TestbedDataset

# ------------------ user settings ------------------
DATASET = 'davis'  # choose 'davis' or 'kiba'
ROOT = 'data'
PROCESSED_DIR = os.path.join(ROOT, 'processed')
MAX_SEQ_LEN = 1000
SEQ_VOC = "ACDEFGHIKLMNPQRSTVWY"
SEQ_DICT = {v: (i + 1) for i, v in enumerate(SEQ_VOC)}
# ---------------------------------------------------


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom):
    """Return a 1D numpy array of atom features (one-hot + degree + Hs + valence + aromatic).
    The result is normalized (L1) to avoid feature-scale issues. """
    symbol_feat = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K',
         'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd',
         'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    )
    degree_feat = one_of_k_encoding(atom.GetDegree(), list(range(11)))
    total_h_feat = one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11)))
    implicit_valence = one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11)))
    aromatic = [atom.GetIsAromatic()]

    feats = symbol_feat + degree_feat + total_h_feat + implicit_valence + aromatic
    arr = np.asarray(feats, dtype=float)
    s = arr.sum()
    if s == 0:
        return arr
    return arr / s


def smiles_to_graph(smiles):
    """Convert SMILES -> (num_atoms, atom_feature_list, edge_index_list)
    edge_index_list: list of [src, dst] (bidirectional)
    Returns None for invalid SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    num_atoms = mol.GetNumAtoms()
    features = [atom_features(a) for a in mol.GetAtoms()]

    edges = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        # add both directions
        edges.append([i, j])
        edges.append([j, i])

    return num_atoms, features, edges


def seq_to_array(seq, max_len=MAX_SEQ_LEN, seq_dict=SEQ_DICT):
    arr = np.zeros(max_len, dtype=int)
    for i, ch in enumerate(seq[:max_len]):
        arr[i] = seq_dict.get(ch, 0)
    return arr


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_deepdta_csv(dataset: str):
    """Read DeepDTA raw files and write simplified train/test CSVs.
    Returns (train_csv_path, test_csv_path)
    """
    fpath = os.path.join(ROOT, dataset)
    train_fold_file = os.path.join(fpath, 'folds', 'train_fold_setting1.txt')
    test_fold_file = os.path.join(fpath, 'folds', 'test_fold_setting1.txt')
    ligands_file = os.path.join(fpath, 'ligands_can.txt')
    proteins_file = os.path.join(fpath, 'proteins.txt')
    y_file = os.path.join(fpath, 'Y')

    for p in (train_fold_file, test_fold_file, ligands_file, proteins_file, y_file):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    train_fold = json.load(open(train_fold_file))
    # train_fold is list of lists; flatten
    train_idx = [i for fold in train_fold for i in fold]
    test_idx = json.load(open(test_fold_file))

    ligands = json.load(open(ligands_file), object_pairs_hook=OrderedDict)
    proteins = json.load(open(proteins_file), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(y_file, 'rb'), encoding='latin1')

    drugs = [Chem.MolToSmiles(Chem.MolFromSmiles(ligands[k]), isomericSmiles=True) for k in ligands.keys()]
    prots = [proteins[k] for k in proteins.keys()]

    affinity = np.asarray(affinity)
    if dataset == 'davis':
        affinity = np.asarray([-np.log10(y / 1e9) for y in affinity])

    # helper to write CSV for given indices
    def write_csv(indices_rows, indices_cols, outpath):
        ensure_dir(os.path.dirname(outpath))
        with open(outpath, 'w') as fw:
            fw.write('compound_iso_smiles,target_sequence,affinity\n')
            for r, c in zip(indices_rows, indices_cols):
                smi = drugs[r]
                seq = prots[c]
                val = affinity[r, c]
                fw.write(f"{smi},{seq},{val}\n")

    # build train/test pairs selection
    rows, cols = np.where(~np.isnan(affinity))
    rows = rows.tolist(); cols = cols.tolist()

    # According to original logic, train pairs are those where both row & col in train_fold indices
    # In original script they selected rows[train_fold], cols[train_fold] (indexing by row index list)
    # We'll follow same behavior: select pairs by index positions.
    train_rows = [rows[i] for i in train_idx]
    train_cols = [cols[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    test_cols = [cols[i] for i in test_idx]

    train_csv = os.path.join(ROOT, f"{dataset}_train.csv")
    test_csv = os.path.join(ROOT, f"{dataset}_test.csv")
    write_csv(train_rows, train_cols, train_csv)
    write_csv(test_rows, test_cols, test_csv)

    return train_csv, test_csv


def build_smile_graphs(smiles_iterable):
    """Return dict: smile -> (num_atoms, features_list, edge_index_list)
    Skips invalid SMILES but keeps a log.
    """
    smile_graph = {}
    bad = []
    for s in tqdm(sorted(set(smiles_iterable)), desc='Converting SMILES'):
        out = smiles_to_graph(s)
        if out is None:
            bad.append(s)
            continue
        smile_graph[s] = out
    if bad:
        print(f"Warning: {len(bad)} invalid SMILES skipped (examples): {bad[:5]}")
    return smile_graph


def prepare_processed(dataset: str):
    train_csv = os.path.join(ROOT, f"{dataset}_train.csv")
    test_csv = os.path.join(ROOT, f"{dataset}_test.csv")
    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        print('Building CSV files from DeepDTA raw files...')
        build_deepdta_csv(dataset)

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # collect unique smiles
    all_smiles = pd.concat([df_train['compound_iso_smiles'], df_test['compound_iso_smiles']]).unique()
    smile_graph = build_smile_graphs(all_smiles)

    # prepare protein arrays (padded/truncated)
    train_prots = [seq_to_array(s) for s in df_train['target_sequence']]
    test_prots = [seq_to_array(s) for s in df_test['target_sequence']]

    train_drugs = df_train['compound_iso_smiles'].tolist()
    test_drugs = df_test['compound_iso_smiles'].tolist()
    train_Y = df_train['affinity'].values
    test_Y = df_test['affinity'].values

    # create processed dataset via TestbedDataset helper (it will save .pt files under data/processed)
    print('Saving processed PyG dataset using TestbedDataset...')
    ensure_dir(PROCESSED_DIR)
    _ = TestbedDataset(root=ROOT, dataset=dataset + '_train', xd=np.array(train_drugs), xt=np.array(train_prots), y=np.array(train_Y), smile_graph=smile_graph)
    _ = TestbedDataset(root=ROOT, dataset=dataset + '_test', xd=np.array(test_drugs), xt=np.array(test_prots), y=np.array(test_Y), smile_graph=smile_graph)

    print('Done. Processed files written to', PROCESSED_DIR)


if __name__ == '__main__':
    prepare_processed(DATASET)


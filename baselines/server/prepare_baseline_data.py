"""Build the shared per-baseline data files for the six-fold CV runs.

Runs on server2 with the mds environment (has rdkit + torch_geometric) from
the project root:

  python baselines/prepare_baseline_data.py

Builds (all in canonical davis_sixfold_all.csv order, i.e. the same index
space as davis_sixfold_all.pt and splits/davis/fold_*.json):

  data/baselines/davis_graphdta_all.pt   78-dim GraphDTA graphs (rdkit)
  data/baselines/davis_deepdtagen_all.pt 94-dim graphs + tokenized drug SMILES
  data/baselines/davis_tokenizer.pkl     DeepDTAGen SMILES tokenizer
"""

import argparse
import json
import os
import pickle
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from collections import deque

import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from tqdm import tqdm


ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OFFICIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "deepdtagen_official")
sys.path.insert(0, OFFICIAL)
from utils import Tokenizer  # noqa: E402


MAX_SEQ_LEN = 1000
SEQ_VOC = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_DICT = {v: (i + 1) for i, v in enumerate(SEQ_VOC)}
MAX_TOKENS = 138  # DeepDTAGen positional encoding supports 138 positions
MAX_PROT_LEN = 1000
MCGEN_HOPS = 3


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features_78(atom):
    feats = (
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
             'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb',
             'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',
             'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'Unknown'])
        + one_of_k_encoding(atom.GetDegree(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11)))
        + [atom.GetIsAromatic()])
    return np.array(feats, dtype=float)


def smile_to_graph_78(smile):
    """Official GraphDTA graph: 78-dim features, two-way directed edges."""
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smile}")
    c_size = mol.GetNumAtoms()
    features = [atom_features_78(a) for a in mol.GetAtoms()]
    features = [f / f.sum() for f in features]
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    import networkx as nx
    g = nx.Graph(edges).to_directed()
    edge_index = [[e1, e2] for e1, e2 in g.edges]
    return c_size, features, edge_index


def _convert_78(smile):
    return smile, smile_to_graph_78(smile)


def seq_cat(prot):
    x = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
    for i, ch in enumerate(prot[:MAX_SEQ_LEN]):
        x[i] = SEQ_DICT.get(ch, 0)
    return x


def get_sample(data, slices, idx):
    sample = data.__class__()
    for key in data.keys():
        item = data[key]
        s = slices[key]
        start, end = int(s[idx]), int(s[idx + 1])
        if key == "edge_index":
            sample[key] = item[:, start:end]
        else:
            sample[key] = item[start:end]
    return sample


def build_graphdta_all(df, out_path):
    unique = sorted(set(df["compound_iso_smiles"]))
    print(f"GraphDTA: converting {len(unique)} unique SMILES to 78-dim graphs ...",
          flush=True)
    smile_graph = {}
    with ProcessPoolExecutor(max_workers=32) as ex:
        for smile, graph in tqdm(
                ex.map(_convert_78, unique, chunksize=10),
                total=len(unique)):
            smile_graph[smile] = graph

    data_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="GraphDTA samples"):
        c_size, features, edge_index = smile_graph[row["compound_iso_smiles"]]
        target = seq_cat(row["target_sequence"])
        d = DATA.Data(
            x=torch.Tensor(features),
            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
            y=torch.FloatTensor([float(row["affinity"])]))
        d.target = torch.LongTensor([target])
        d.c_size = torch.LongTensor([c_size])
        data_list.append(d)
    data, slices = InMemoryDataset.collate(data_list)
    del data_list
    torch.save((data, slices), out_path)
    print(f"GraphDTA all-file written: {out_path}", flush=True)


def build_deepdtagen_all(df, base_pt_path, out_path, tokenizer_path):
    data, slices = torch.load(base_pt_path, map_location="cpu",
                              weights_only=False)
    n = int(slices["y"][-1].item())
    if n != len(df):
        raise ValueError(f"sample mismatch: {n} != {len(df)}")

    all_smiles = set(df["compound_iso_smiles"])
    tokenizer = Tokenizer(Tokenizer.gen_vocabs(all_smiles))
    with open(tokenizer_path, "wb") as fh:
        pickle.dump(tokenizer, fh)
    print(f"Tokenizer vocab size: {len(tokenizer)}", flush=True)

    token_seqs = []
    max_len = 0
    for smile in tqdm(df["compound_iso_smiles"], desc="Tokenize SMILES"):
        toks = tokenizer.parse(smile)
        if len(toks) > MAX_TOKENS:
            toks = toks[:MAX_TOKENS]
        max_len = max(max_len, len(toks))
        token_seqs.append(toks)
    print(f"Max tokenized SMILES length: {max_len} (cap {MAX_TOKENS})",
          flush=True)

    pad = tokenizer.s2i["<pad>"]
    data_list = []
    for i in tqdm(range(n), desc="DeepDTAGen samples"):
        d = get_sample(data, slices, i)
        toks = token_seqs[i]
        arr = np.full((1, max_len), pad, dtype=np.int64)
        arr[0, :len(toks)] = toks
        d.target_seq = torch.LongTensor(arr)
        data_list.append(d)
    out_data, out_slices = InMemoryDataset.collate(data_list)
    del data_list
    torch.save((out_data, out_slices), out_path)
    print(f"DeepDTAGen all-file written: {out_path}", flush=True)


def exact_distance_edges(n, edge_pairs, k):
    """Undirected edges between nodes at exact graph distance k."""
    adj = [[] for _ in range(n)]
    for u, v in edge_pairs:
        adj[u].append(v)
        adj[v].append(u)
    edges = []
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for t in range(n):
            if dist[t] == k:
                edges.append((s, t))
    return edges


def smile_to_int_graph(smile, atom_vocab):
    """Atom-integer graph (no explicit bond type) + 1..3-hop edge lists."""
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smile}")
    n = mol.GetNumAtoms()
    x = [atom_vocab.get(a.GetSymbol(), 0) for a in mol.GetAtoms()]
    edge_pairs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                  for b in mol.GetBonds()]
    hops = []
    for k in range(1, MCGEN_HOPS + 1):
        edges = exact_distance_edges(n, edge_pairs, k)
        hops.append([[u, v] for u, v in edges])
    return n, x, hops


def build_gdilateddta_all(df, out_path, meta_path):
    unique = sorted(set(df["compound_iso_smiles"]))
    symbols = set()
    for smile in unique:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError(f"invalid SMILES: {smile}")
        symbols.update(a.GetSymbol() for a in mol.GetAtoms())
    atom_vocab = {sym: i + 1 for i, sym in enumerate(sorted(symbols))}
    print(f"GDilatedDTA: {len(unique)} unique SMILES, "
          f"atom vocab size={len(atom_vocab)}", flush=True)

    def edge_tensor(hop_list):
        if hop_list:
            return torch.LongTensor(hop_list).transpose(1, 0)
        return torch.empty((2, 0), dtype=torch.long)

    smile_graph = {}
    for smile in tqdm(unique, desc="GDilatedDTA graphs"):
        smile_graph[smile] = smile_to_int_graph(smile, atom_vocab)

    data_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="GDilatedDTA samples"):
        c_size, x, hops = smile_graph[row["compound_iso_smiles"]]
        protein = seq_cat(row["target_sequence"])
        d = DATA.Data(
            x=torch.LongTensor(x),
            edge_index=edge_tensor(hops[0]),
            edge_index2=edge_tensor(hops[1]),
            edge_index3=edge_tensor(hops[2]),
            protein=torch.LongTensor(protein).unsqueeze(0),
            y=torch.FloatTensor([float(row["affinity"])]))
        d.c_size = torch.LongTensor([c_size])
        data_list.append(d)
    data, slices = InMemoryDataset.collate(data_list)
    del data_list
    torch.save((data, slices), out_path)
    meta = {
        "atom_vocab": atom_vocab,
        "atom_vocab_size": len(atom_vocab) + 1,
        "max_protein_len": MAX_PROT_LEN,
        "hops": MCGEN_HOPS,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"GDilatedDTA all-file written: {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="davis")
    args = parser.parse_args()
    ds = args.dataset
    os.makedirs(os.path.join(ROOT, "data", "baselines"), exist_ok=True)
    df = pd.read_csv(os.path.join(ROOT, "data", f"{ds}_sixfold_all.csv"))
    print(f"Loaded {len(df)} rows from {ds}_sixfold_all.csv", flush=True)
    build_graphdta_all(df, os.path.join(ROOT, "data", "baselines",
                                        f"{ds}_graphdta_all.pt"))
    build_deepdtagen_all(
        df,
        os.path.join(ROOT, "data", "processed", f"{ds}_sixfold_all.pt"),
        os.path.join(ROOT, "data", "baselines", f"{ds}_deepdtagen_all.pt"),
        os.path.join(ROOT, "data", "baselines", f"{ds}_tokenizer.pkl"))
    build_gdilateddta_all(
        df,
        os.path.join(ROOT, "data", "baselines", f"{ds}_gdilateddta_all.pt"),
        os.path.join(ROOT, "data", "baselines", f"{ds}_gdilateddta_meta.json"))
    print("Done.")


if __name__ == "__main__":
    main()

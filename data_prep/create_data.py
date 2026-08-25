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

import argparse
import os
import json
import pickle
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem

from utils import TestbedDataset

# ------------------ user settings ------------------
DATASET = 'pAAP_y'  # choose 'davis','kiba','bindingdb'
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
    """Return the 94-dimensional atom feature vector used by MDS.

    Features comprise atom type (44), degree (11), total hydrogens (11),
    implicit valence (11), formal charge (11), hybridization (5), and
    aromaticity (1). Features are kept as binary 0/1 values (no L1
    normalization): the one-hot structure is information-bearing and the
    network input projection can absorb the scale.
    """
    symbol_feat = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K',
         'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd',
         'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    )
    degree_feat = one_of_k_encoding(atom.GetDegree(), list(range(11)))
    total_h_feat = one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11)))
    implicit_valence = one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11)))
    charge_feat = one_of_k_encoding_unk(atom.GetFormalCharge(), list(range(-5, 6)))
    hybridization_types = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ]
    hybridization_feat = [atom.GetHybridization() == value for value in hybridization_types]
    aromatic = [atom.GetIsAromatic()]

    feats = (
        symbol_feat
        + degree_feat
        + total_h_feat
        + implicit_valence
        + charge_feat
        + hybridization_feat
        + aromatic
    )
    arr = np.asarray(feats, dtype=float)
    return arr


def smiles_to_graph(smiles):
    """Convert SMILES -> (num_atoms, atom_feature_list, edge_index_list, edge_attr_list)
    edge_index_list: list of [src, dst] (bidirectional)
    edge_attr_list: integer bond type per directed edge
        (0 single, 1 double, 2 triple, 3 aromatic, 4 other)
    Returns None for invalid SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    num_atoms = mol.GetNumAtoms()
    features = [atom_features(a) for a in mol.GetAtoms()]

    edges = []
    edge_attrs = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        bond_type = b.GetBondType()
        if bond_type == rdchem.BondType.SINGLE:
            btype = 0
        elif bond_type == rdchem.BondType.DOUBLE:
            btype = 1
        elif bond_type == rdchem.BondType.TRIPLE:
            btype = 2
        elif bond_type == rdchem.BondType.AROMATIC:
            btype = 3
        else:
            btype = 4
        # add both directions
        edges.append([i, j])
        edges.append([j, i])
        edge_attrs.append(btype)
        edge_attrs.append(btype)

    return num_atoms, features, edges, edge_attrs


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


def _canonical_or_seeded_six_folds(dataset: str, root: str, seed: int):
    """Return six disjoint folds in processed-dataset index space.

    Davis and KIBA retain the five published DeepDTA training folds and use
    the published held-out fold as fold 5. Their processed files are ordered
    as flattened training folds followed by the held-out fold, so the mapping
    to the concatenated ``*_train`` + ``*_test`` datasets is exact.

    BindingDB has no raw DeepDTA fold files in this project snapshot. Its
    existing processed train/test samples are therefore pooled and assigned
    once to six deterministic, size-balanced pair-level folds.
    """
    dataset_dir = os.path.join(root, dataset)
    train_fold_file = os.path.join(dataset_dir, 'folds', 'train_fold_setting1.txt')
    test_fold_file = os.path.join(dataset_dir, 'folds', 'test_fold_setting1.txt')

    if os.path.exists(train_fold_file) and os.path.exists(test_fold_file):
        with open(train_fold_file) as fh:
            source_folds = json.load(fh)
        with open(test_fold_file) as fh:
            held_out = json.load(fh)
        if len(source_folds) != 5:
            raise ValueError(f'{dataset}: expected five training folds, got {len(source_folds)}')
        source_folds = [list(map(int, fold)) for fold in source_folds] + [list(map(int, held_out))]
        source_order = [idx for fold in source_folds for idx in fold]
        if len(source_order) != len(set(source_order)):
            raise ValueError(f'{dataset}: duplicate pair indices in canonical folds')
        source_to_processed = {source_idx: pos for pos, source_idx in enumerate(source_order)}
        processed_folds = [sorted(source_to_processed[idx] for idx in fold) for fold in source_folds]
        provenance = 'canonical_deepdta_5_train_folds_plus_held_out_fold'
    else:
        train_data = TestbedDataset(root=root, dataset=dataset + '_train')
        test_data = TestbedDataset(root=root, dataset=dataset + '_test')
        n_samples = len(train_data) + len(test_data)
        rng = np.random.default_rng(seed)
        processed_folds = [sorted(chunk.tolist()) for chunk in np.array_split(rng.permutation(n_samples), 6)]
        source_folds = None
        provenance = 'deterministic_seeded_balanced_pair_folds_from_processed_train_plus_test'

    return processed_folds, source_folds, provenance


def generate_sixfold_manifests(datasets, root: str, output_dir: str, seed: int = 42):
    """Create exact 4/1/1 train/validation/test indices for six-fold CV.

    In outer run i, fold i is the one-time test set, fold (i+1) mod 6 is the
    validation set, and the other four folds are training data. Thus every
    sample is test data exactly once and validation data exactly once, and no
    test sample participates in scheduling, early stopping, or checkpointing.
    """
    os.makedirs(output_dir, exist_ok=True)
    summaries = []
    for dataset in datasets:
        dataset = dataset.lower()
        fold_members, source_folds, provenance = _canonical_or_seeded_six_folds(dataset, root, seed)
        flat = [idx for fold in fold_members for idx in fold]
        if len(flat) != len(set(flat)) or set(flat) != set(range(len(flat))):
            raise ValueError(f'{dataset}: folds are not a disjoint complete partition')

        dataset_out = os.path.join(output_dir, dataset)
        os.makedirs(dataset_out, exist_ok=True)
        fold_sizes = [len(fold) for fold in fold_members]
        membership = {
            'schema_version': 1,
            'dataset': dataset,
            'seed': seed,
            'protocol': 'six_fold_pairwise_cv_with_cyclic_validation',
            'index_space': 'concatenated_processed_train_then_test',
            'provenance': provenance,
            'n_samples': len(flat),
            'fold_sizes': fold_sizes,
            'fold_members': fold_members,
        }
        if source_folds is not None:
            membership['source_pair_index_space'] = 'row_major_non_nan_affinity_pairs'
            membership['source_pair_fold_members'] = source_folds
        with open(os.path.join(dataset_out, 'fold_membership.json'), 'w') as fh:
            json.dump(membership, fh, indent=2)

        test_counts = np.zeros(len(flat), dtype=np.int8)
        val_counts = np.zeros(len(flat), dtype=np.int8)
        for fold_id in range(6):
            test_fold = fold_id
            validation_fold = (fold_id + 1) % 6
            train_folds = [i for i in range(6) if i not in (test_fold, validation_fold)]
            train_indices = sorted(idx for i in train_folds for idx in fold_members[i])
            validation_indices = fold_members[validation_fold]
            test_indices = fold_members[test_fold]
            test_counts[test_indices] += 1
            val_counts[validation_indices] += 1
            split = {
                'schema_version': 1,
                'dataset': dataset,
                'base_seed': seed,
                'run_seed': seed + fold_id,
                'outer_fold': fold_id,
                'train_fold_ids': train_folds,
                'validation_fold_id': validation_fold,
                'test_fold_id': test_fold,
                'train_indices': train_indices,
                'validation_indices': validation_indices,
                'test_indices': test_indices,
                'sizes': {
                    'train': len(train_indices),
                    'validation': len(validation_indices),
                    'test': len(test_indices),
                },
            }
            with open(os.path.join(dataset_out, f'fold_{fold_id}.json'), 'w') as fh:
                json.dump(split, fh, indent=2)

        if not (np.all(test_counts == 1) and np.all(val_counts == 1)):
            raise AssertionError(f'{dataset}: each sample must be test and validation exactly once')
        summary = {
            'dataset': dataset,
            'n_samples': len(flat),
            'fold_sizes': fold_sizes,
            'provenance': provenance,
            'audit': {
                'complete_coverage': True,
                'pairwise_disjoint_folds': True,
                'each_sample_test_once': True,
                'each_sample_validation_once': True,
                'train_validation_test_disjoint_per_run': True,
            },
        }
        with open(os.path.join(dataset_out, 'audit_summary.json'), 'w') as fh:
            json.dump(summary, fh, indent=2)
        # The time-constrained primary experiment uses one fixed 4/1/1 split:
        # parts 2-5 train, part 1 validates, and part 0 tests. Rotation files
        # remain available for a future full six-run cross-validation study.
        fixed_split_path = os.path.join(dataset_out, 'fixed_six_part_split.json')
        with open(os.path.join(dataset_out, 'fold_0.json')) as source_fh:
            fixed_split = json.load(source_fh)
        fixed_split['executed_protocol'] = 'one_fixed_six_part_holdout_run_not_rotated_cross_validation'
        with open(fixed_split_path, 'w') as fh:
            json.dump(fixed_split, fh, indent=2)
        summaries.append(summary)
        print(f"{dataset}: six-fold manifests written to {dataset_out}; sizes={fold_sizes}")
    return summaries


def prepare_sixfold_processed(dataset: str, root: str, force_rebuild: bool = False):
    """Build a fresh all-sample PyG file in canonical six-fold order.

    This deliberately uses a new ``*_sixfold_all.pt`` name so stale legacy
    train/test artifacts are never overwritten or silently reused.
    """
    dataset = dataset.lower()
    fpath = os.path.join(root, dataset)
    required = [
        os.path.join(fpath, 'folds', 'train_fold_setting1.txt'),
        os.path.join(fpath, 'folds', 'test_fold_setting1.txt'),
        os.path.join(fpath, 'ligands_can.txt'),
        os.path.join(fpath, 'proteins.txt'),
        os.path.join(fpath, 'Y'),
    ]
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f'{dataset}: six-fold raw source not found: {path}')
    with open(required[0]) as fh:
        source_folds = json.load(fh)
    with open(required[1]) as fh:
        held_out = json.load(fh)
    source_order = [int(idx) for fold in source_folds for idx in fold] + [int(idx) for idx in held_out]
    if len(source_order) != len(set(source_order)):
        raise ValueError(f'{dataset}: duplicate source indices')

    with open(required[2]) as fh:
        ligands = json.load(fh, object_pairs_hook=OrderedDict)
    with open(required[3]) as fh:
        proteins = json.load(fh, object_pairs_hook=OrderedDict)
    with open(required[4], 'rb') as fh:
        affinity = np.asarray(pickle.load(fh, encoding='latin1'))
    drugs = []
    for value in ligands.values():
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f'{dataset}: invalid ligand SMILES: {value}')
        drugs.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    prots = list(proteins.values())
    if dataset == 'davis':
        affinity = -np.log10(affinity / 1e9)
    rows, cols = np.where(~np.isnan(affinity))
    if set(source_order) != set(range(len(rows))):
        raise ValueError(f'{dataset}: canonical folds do not cover all non-missing affinity pairs')

    ordered_drugs = np.asarray([drugs[rows[idx]] for idx in source_order])
    ordered_sequences = [prots[cols[idx]] for idx in source_order]
    ordered_targets = np.asarray([seq_to_array(seq) for seq in ordered_sequences])
    ordered_y = np.asarray([affinity[rows[idx], cols[idx]] for idx in source_order], dtype=float)
    smile_graph = build_smile_graphs(ordered_drugs)

    csv_path = os.path.join(root, f'{dataset}_sixfold_all.csv')
    pd.DataFrame({
        'source_pair_index': source_order,
        'compound_iso_smiles': ordered_drugs,
        'target_sequence': ordered_sequences,
        'affinity': ordered_y,
    }).to_csv(csv_path, index=False)
    processed_path = os.path.join(root, 'processed', f'{dataset}_sixfold_all.pt')
    if os.path.exists(processed_path):
        if force_rebuild:
            print(f'--force-rebuild: removing existing six-fold artifact: {processed_path}')
            os.remove(processed_path)
        else:
            raise FileExistsError(
                f'Refusing to reuse/overwrite existing six-fold artifact: '
                f'{processed_path} (pass --force-rebuild to regenerate)')
    TestbedDataset(
        root=root, dataset=dataset + '_sixfold_all', xd=ordered_drugs,
        xt=ordered_targets, y=ordered_y, smile_graph=smile_graph)
    print(f'{dataset}: fresh six-fold data written to {csv_path} and {processed_path}')
    return processed_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare MDS data or generate strict six-fold CV indices.')
    parser.add_argument('--dataset', default=DATASET, help='Dataset for legacy preprocessing mode.')
    parser.add_argument('--root', default=ROOT, help='Data root containing raw and/or processed datasets.')
    parser.add_argument('--make-sixfold', action='store_true', help='Generate exact six-fold train/validation/test indices.')
    parser.add_argument('--prepare-sixfold-data', action='store_true', help='Build fresh canonical *_sixfold_all.pt data.')
    parser.add_argument('--datasets', nargs='+', default=['davis', 'kiba', 'bindingdb'])
    parser.add_argument('--split-output', default='data/splits')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force-rebuild', action='store_true',
                        help='Delete an existing *_sixfold_all.pt before rebuilding '
                             '(use with --prepare-sixfold-data).')
    args = parser.parse_args()
    ROOT = args.root
    PROCESSED_DIR = os.path.join(ROOT, 'processed')
    if args.prepare_sixfold_data:
        prepare_sixfold_processed(args.dataset, args.root, force_rebuild=args.force_rebuild)
    elif args.make_sixfold:
        generate_sixfold_manifests(args.datasets, args.root, args.split_output, args.seed)
    else:
        prepare_processed(args.dataset)

"""Build and validate the *_sixfold_all.pt files used by train.py.

The fixed six-part holdout splits live in splits/<dataset>/ and index the
canonical sample order of <dataset>_sixfold_all.pt. This script guarantees
that file exists, has the right sample count, and uses the current 94-dim
atom features expected by the model.

Per-dataset strategy:
  davis / kiba   rebuild from the raw DeepDTA files (data/<dataset>/folds,
                 ligands_can.txt, proteins.txt, Y) in canonical fold order.
                 This is required because legacy processed files may use an
                 older 78-dim feature encoding or a different sample order.
  bindingdb      no raw DeepDTA fold files exist; the split manifests were
                 generated over the processed train+test files, so the
                 sixfold file is exactly bindingdb_train.pt + bindingdb_test.pt.

Usage (from the project root):
  python prepare_sixfold_data.py --datasets kiba bindingdb
  python prepare_sixfold_data.py --datasets kiba --output-dir /root/autodl-tmp/mds_data
"""

import argparse
import json
import os
import shutil

import torch
from torch_geometric.data import InMemoryDataset

import create_data
from utils import TestbedDataset


EXPECTED_ATOM_DIM = 94


def split_total(dataset):
    path = os.path.join("splits", dataset, "fixed_six_part_split.json")
    with open(path, encoding="utf-8") as fh:
        split = json.load(fh)
    return sum(split["sizes"].values())


def verify(dataset, root):
    path = os.path.join(root, "processed", dataset + "_sixfold_all.pt")
    if not os.path.exists(path):
        return False, "missing"
    data, slices = torch.load(path, map_location="cpu", weights_only=False)
    n = slices["y"][-1].item()
    total = split_total(dataset)
    if n != total:
        return False, "sample count %d != split total %d" % (n, total)
    x_dim = data["x"].shape[1]
    if x_dim != EXPECTED_ATOM_DIM:
        return False, "atom feature dim %d != expected %d" % (x_dim, EXPECTED_ATOM_DIM)
    return True, "%d samples, %d-dim" % (n, x_dim)


def has_raw(dataset, root):
    return all(os.path.exists(os.path.join(root, dataset, "folds", name)) for name in
               ("train_fold_setting1.txt", "test_fold_setting1.txt"))


def remove_sixfold(dataset, root, output_dir):
    for path in (
            os.path.join(root, "processed", dataset + "_sixfold_all.pt"),
            os.path.join(output_dir, dataset + "_sixfold_all.pt")):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def link_into_processed(dataset, root, dest):
    link = os.path.join(root, "processed", dataset + "_sixfold_all.pt")
    if os.path.abspath(link) == os.path.abspath(dest):
        return link
    try:
        os.remove(link)
    except FileNotFoundError:
        pass
    os.symlink(dest, link)
    return link


def build(dataset, root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if has_raw(dataset, root):
        print(f"{dataset}: rebuilding from raw DeepDTA files ...")
        create_data.prepare_sixfold_processed(dataset, root)
        src = os.path.join(root, "processed", dataset + "_sixfold_all.pt")
    else:
        print(f"{dataset}: concatenating processed train/test files ...")
        train = TestbedDataset(root=root, dataset=dataset + "_train")
        test = TestbedDataset(root=root, dataset=dataset + "_test")
        data_list = [d for d in train] + [d for d in test]
        data, slices = InMemoryDataset.collate(data_list)
        del data_list
        src = os.path.join(root, "processed", dataset + "_sixfold_all.pt")
        torch.save((data, slices), src)

    dest = os.path.join(output_dir, dataset + "_sixfold_all.pt")
    if os.path.abspath(src) != os.path.abspath(dest):
        shutil.move(src, dest)
        link = link_into_processed(dataset, root, dest)
    else:
        link = src

    ok, msg = verify(dataset, root)
    if not ok:
        raise RuntimeError(f"{dataset}: build failed verification ({msg})")
    print(f"{dataset}: OK ({msg}) -> {link}")


def main():
    parser = argparse.ArgumentParser(
        description="Build/validate sixfold all-sample data files.")
    parser.add_argument("--datasets", nargs="+", default=["kiba", "bindingdb"])
    parser.add_argument("--root", default="data")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for the sixfold .pt files "
                             "(default: <root>/processed)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.root, "processed")
    for dataset in args.datasets:
        dataset = dataset.lower()
        ok, msg = verify(dataset, args.root)
        if ok:
            print(f"{dataset}: already valid, skipping ({msg})")
            continue
        print(f"{dataset}: invalid or missing ({msg}), rebuilding ...")
        remove_sixfold(dataset, args.root, output_dir)
        build(dataset, args.root, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()

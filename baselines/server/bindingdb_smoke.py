"""CPU-only smoke test: every baseline consumes the bindingdb data."""

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/root/mds")
sys.path.insert(0, "/root/mds/baselines/deepdtagen_official")

import torch

from baselines.common import SavedGraphDataset, load_rows, load_split


def check_graphdta():
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from baselines.graphdta_baseline import GCNNet
    ds = SavedGraphDataset("/root/mds/data/baselines/bindingdb_graphdta_all.pt")
    loader = DataLoader(Subset(ds, list(range(32))), batch_size=8)
    model = GCNNet()
    batch = next(iter(loader))
    loss = torch.nn.functional.mse_loss(model(batch), batch.y.view(-1, 1).float())
    loss.backward()
    print("graphdta: OK loss=%.4f" % float(loss), flush=True)


def check_gdilateddta():
    import json
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from baselines.gdilateddta_baseline import GDilatedDTAModel
    meta = json.load(open("/root/mds/data/baselines/bindingdb_gdilateddta_meta.json"))
    ds = SavedGraphDataset("/root/mds/data/baselines/bindingdb_gdilateddta_all.pt")
    loader = DataLoader(Subset(ds, list(range(32))), batch_size=8)
    model = GDilatedDTAModel(int(meta["atom_vocab_size"]))
    batch = next(iter(loader))
    loss = torch.nn.functional.mse_loss(model(batch), batch.y.view(-1, 1).float())
    loss.backward()
    print("gdilateddta: OK loss=%.4f" % float(loss), flush=True)


def check_deepdtagen():
    import pickle
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from utils import Tokenizer
    from model import DeepDTAGen
    with open("/root/mds/data/baselines/bindingdb_tokenizer.pkl", "rb") as fh:
        tokenizer = pickle.load(fh)
    ds = SavedGraphDataset("/root/mds/data/baselines/bindingdb_deepdtagen_all.pt")
    loader = DataLoader(Subset(ds, list(range(8))), batch_size=4)
    model = DeepDTAGen(tokenizer)
    batch = next(iter(loader))
    pred, _, lm_loss, kl_loss = model(batch)
    loss = torch.nn.functional.mse_loss(pred, batch.y.view(-1, 1).float())
    loss.backward()
    print("deepdtagen: OK loss=%.4f" % float(loss), flush=True)


def check_widedta():
    from baselines.widedta_baseline import (
        WideDTAModel, build_vocab, encode_frame)
    df = load_rows("bindingdb")
    drug_vocab, prot_vocab = build_vocab(df)
    x, p, y = encode_frame(df, list(range(32)), drug_vocab, prot_vocab)
    model = WideDTAModel(len(drug_vocab) + 1, len(prot_vocab) + 1)
    loss = torch.nn.functional.mse_loss(model(x, p), y.view(-1, 1))
    loss.backward()
    print("widedta: OK loss=%.4f" % float(loss), flush=True)


def check_ssmdta():
    from fairseq.data import Dictionary
    from baselines.ssmdta_baseline import build_model, encode_frame, pad_collate
    from torch.utils.data import DataLoader
    df = load_rows("bindingdb")
    mol_dict = Dictionary.load("/root/mds/baselines/ssmdta_official/dict.mol.txt")
    pro_dict = Dictionary.load("/root/mds/baselines/ssmdta_official/dict.pro.txt")
    mols, prots, ys = encode_frame(df, list(range(8)), mol_dict, pro_dict)
    loader = DataLoader(list(zip(mols, prots, ys)), batch_size=4,
                        collate_fn=pad_collate)
    model, _ = build_model(mol_dict, pro_dict, encoder_layers=2)
    mol, prot, label = next(iter(loader))
    out = model(src_tokens_0=mol, src_tokens_1=prot, features_only=True,
                classification_head_name="sentence_classification_head")[0]
    loss = torch.nn.functional.mse_loss(out.view(-1), label)
    loss.backward()
    print("ssmdta: OK loss=%.4f" % float(loss), flush=True)


if __name__ == "__main__":
    check_graphdta()
    check_gdilateddta()
    check_deepdtagen()
    check_widedta()
    check_ssmdta()
    print("BINDINGDB SMOKE OK")

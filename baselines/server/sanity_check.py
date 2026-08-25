"""Quick single-batch sanity check for the four baseline data paths.

Runs a couple of forward/backward steps on the GPU without any checkpointing,
just to confirm each model can consume the davis six-fold data:

  python baselines/server/sanity_check.py
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baselines"))
sys.path.insert(0, os.path.join(ROOT, "baselines", "deepdtagen_official"))

import torch

from baselines.common import SavedGraphDataset, load_rows


def check_graphdta():
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from baselines.graphdta_baseline import GCNNet

    ds = SavedGraphDataset(os.path.join(ROOT, "data", "baselines", "davis_graphdta_all.pt"))
    loader = DataLoader(Subset(ds, list(range(64))), batch_size=16, shuffle=True)
    model = GCNNet().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    batch = next(iter(loader)).cuda()
    loss = torch.nn.functional.mse_loss(model(batch), batch.y.view(-1, 1).float())
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("GraphDTA GCN batch OK, loss=%.4f" % float(loss), flush=True)


def check_attentiondta():
    from torch.utils.data import DataLoader, TensorDataset
    from baselines.attentiondta_baseline import AttentionDTA, encode_frame

    df = load_rows("davis")
    x, p, y = encode_frame(df, list(range(64)))
    loader = DataLoader(TensorDataset(x, p, y), batch_size=16)
    model = AttentionDTA().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    drug, prot, label = next(iter(loader))
    loss = torch.nn.functional.mse_loss(model(drug.cuda(), prot.cuda()),
                                        label.view(-1, 1).cuda())
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("AttentionDTA batch OK, loss=%.4f" % float(loss), flush=True)


def check_deepdtagen():
    import pickle
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from utils import Tokenizer
    from model import DeepDTAGen
    from FetterGrad import FetterGrad

    with open(os.path.join(ROOT, "data", "baselines", "davis_tokenizer.pkl"), "rb") as fh:
        tokenizer = pickle.load(fh)
    ds = SavedGraphDataset(os.path.join(ROOT, "data", "baselines", "davis_deepdtagen_all.pt"))
    loader = DataLoader(Subset(ds, list(range(16))), batch_size=4, shuffle=True)
    model = DeepDTAGen(tokenizer).cuda()
    optimizer = FetterGrad(torch.optim.Adam(model.parameters(), lr=2e-4))
    mse_f = torch.nn.MSELoss()
    batch = next(iter(loader)).cuda()
    pred, _, lm_loss, kl_loss = model(batch)
    mse_loss = mse_f(pred, batch.y.view(-1, 1).float())
    loss = kl_loss * 0.001 + mse_loss + lm_loss
    optimizer.ft_backward([loss, mse_loss])
    optimizer.step()
    print("DeepDTAGen batch OK, loss=%.4f (mse %.4f, lm %.4f, kl %.4f)"
          % (float(loss), float(mse_loss), float(lm_loss), float(kl_loss)),
          flush=True)


def check_widedta():
    import deepsmiles  # noqa: F401
    from torch.utils.data import DataLoader, TensorDataset
    from baselines.widedta_baseline import (
        WideDTAModel, build_vocab, encode_frame)

    df = load_rows("davis")
    drug_vocab, prot_vocab = build_vocab(df)
    x, p, y = encode_frame(df, list(range(64)), drug_vocab, prot_vocab)
    loader = DataLoader(TensorDataset(x, p, y), batch_size=16)
    model = WideDTAModel(len(drug_vocab) + 1, len(prot_vocab) + 1).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    drug, prot, label = next(iter(loader))
    loss = torch.nn.functional.mse_loss(
        model(drug.cuda(), prot.cuda()), label.view(-1, 1).cuda())
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("WideDTA batch OK (drug vocab %d, prot vocab %d), loss=%.4f"
          % (len(drug_vocab), len(prot_vocab), float(loss)), flush=True)


def check_gdilateddta():
    import json
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from baselines.gdilateddta_baseline import GDilatedDTAModel

    meta = json.load(open(os.path.join(ROOT, "data", "baselines", "davis_gdilateddta_meta.json")))
    ds = SavedGraphDataset(os.path.join(ROOT, "data", "baselines", "davis_gdilateddta_all.pt"))
    loader = DataLoader(Subset(ds, list(range(64))), batch_size=16, shuffle=True)
    model = GDilatedDTAModel(int(meta["atom_vocab_size"])).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    batch = next(iter(loader)).cuda()
    loss = torch.nn.functional.mse_loss(model(batch), batch.y.view(-1, 1).float())
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("GDilatedDTA batch OK, loss=%.4f" % float(loss), flush=True)


def check_ssmdta():
    from fairseq.data import Dictionary
    from baselines.ssmdta_baseline import (
        build_model, encode_frame, pad_collate)
    from torch.utils.data import DataLoader

    df = load_rows("davis")
    mol_dict = Dictionary.load(os.path.join(ROOT, "baselines", "ssmdta_official", "dict.mol.txt"))
    pro_dict = Dictionary.load(os.path.join(ROOT, "baselines", "ssmdta_official", "dict.pro.txt"))
    mols, prots, ys = encode_frame(df, list(range(8)), mol_dict, pro_dict)
    loader = DataLoader(list(zip(mols, prots, ys)), batch_size=4,
                        collate_fn=pad_collate)
    model, _ = build_model(mol_dict, pro_dict, encoder_layers=2)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    mol, prot, label = next(iter(loader))
    mol, prot, label = mol.cuda(), prot.cuda(), label.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, _ = model(src_tokens_0=mol, src_tokens_1=prot, features_only=True,
                       classification_head_name="sentence_classification_head")
        loss = torch.nn.functional.mse_loss(out.float().view(-1), label)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("SSM-DTA batch OK, loss=%.4f" % float(loss), flush=True)


if __name__ == "__main__":
    check_graphdta()
    check_attentiondta()
    check_deepdtagen()
    check_widedta()
    check_gdilateddta()
    check_ssmdta()
    print("ALL SANITY CHECKS PASSED")

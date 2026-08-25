"""Extract per-residue ESM-2 features for a sixfold dataset and cache to disk.

This is the one-time offline step that makes arm D (enhanced) trainable at
the same speed as arm C: ESM is frozen and the features depend only on the
sequences, so we compute them once and let training read the cache instead of
running ESM on every batch.

Usage (run on the training server, e.g. Server 1):
  python extract_esm_features.py --dataset davis \
      --esm-tag esm2_t33_650M_UR50D --batch-size 64 \
      --out /root/autodl-tmp/mds_data/davis_esm_t33.pt

Output (torch.save):
  {
    "features": FloatTensor[N, max_len, proj_dim]  (fp16),
    "esm_tag": str, "proj_dim": int, "max_len": int,
  }

``features[i]`` is aligned with the i-th sample of
``data/processed/<dataset>_sixfold_all.pt`` (same canonical order), padded to
``max_len`` (1000).  The raw 1280-dim ESM output is projected to ``proj_dim``
with a FIXED random-initialised projection (default init, frozen), which is
equivalent to freezing the ESM projection layer; this keeps the cache at
~15 GB (256-dim, fp16) instead of ~77 GB (1280-dim, fp16).
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn


ESM_TAG_LAYER = {
    "esm2_t33_650M_UR50D": 33,
    "esm2_t30_150M_UR50D": 30,
    "esm2_t12_35M_UR50D": 12,
    "esm2_t6_8M_UR50D": 6,
}

ESM_TAG_DIM = {
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t6_8M_UR50D": 320,
}

_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def load_esm(esm_tag, device):
    loaded = torch.hub.load("facebookresearch/esm:main", esm_tag)
    if isinstance(loaded, tuple):
        model, alphabet = loaded
    else:
        model, alphabet = loaded, loaded.alphabet
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    model.to(device)
    return model, alphabet


def seqs_from_tokens(tokens_batch):
    """Convert an integer token batch [B, L] into ESM sequence strings."""
    seqs = []
    for row in tokens_batch:
        chars = []
        for token_id in row:
            if token_id == 0:
                break  # padding is trailing
            chars.append(_AA_ALPHABET[token_id - 1] if 1 <= token_id <= 20 else "X")
        seqs.append("".join(chars))
    return seqs


def main():
    parser = argparse.ArgumentParser(description="Cache frozen ESM-2 residue features.")
    parser.add_argument("--dataset", default="davis")
    parser.add_argument("--root", default="data",
                        help="Directory containing processed/<dataset>_sixfold_all.pt")
    parser.add_argument("--esm-tag", default="esm2_t33_650M_UR50D")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--out", default="data/processed/davis_esm_t33.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process the first N samples (0 = all); "
                             "useful for smoke tests.")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import TestbedDataset

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = TestbedDataset(root=args.root, dataset=args.dataset + "_sixfold_all")
    tokens = dataset.data.target  # [N, L] long (collated)
    if args.limit > 0:
        tokens = tokens[:args.limit]
    n_samples, max_len = tokens.shape
    print(f"samples={n_samples} max_len={max_len} device={device}", flush=True)

    model, alphabet = load_esm(args.esm_tag, device)
    repr_layer = ESM_TAG_LAYER[args.esm_tag]
    proj = nn.Linear(ESM_TAG_DIM[args.esm_tag], args.proj_dim)
    proj = proj.to(device)
    proj.eval()

    features = torch.zeros(
        n_samples, max_len, args.proj_dim, dtype=torch.float16, device="cpu")
    lengths = (tokens != 0).sum(dim=1).tolist()

    start = time.time()
    done = 0
    for i in range(0, n_samples, args.batch_size):
        batch_tokens = tokens[i:i + args.batch_size]
        seqs = seqs_from_tokens(batch_tokens.tolist())
        _, _, batched = alphabet.get_batch_converter()(
            [("", s) for s in seqs])
        batched = batched.to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(batched, repr_layers=[repr_layer])
                reps = out["representations"][repr_layer][:, 1:-1]  # [B, L_i, D]
            proj_reps = proj(reps.float()).half().cpu()
        for j in range(batch_tokens.shape[0]):
            n = lengths[i + j]
            features[i + j, :n] = proj_reps[j, :n]
        done += batch_tokens.shape[0]
        if done % (args.batch_size * 20) == 0 or done == n_samples:
            elapsed = time.time() - start
            eta = elapsed / done * (n_samples - done) if done else 0
            print(
                f"{done}/{n_samples} ({100.0 * done / n_samples:.1f}%) "
                f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
                flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({
        "features": features,
        "esm_tag": args.esm_tag,
        "proj_dim": args.proj_dim,
        "max_len": max_len,
    }, args.out)
    print(f"saved to {args.out} "
          f"({os.path.getsize(args.out) / 1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()

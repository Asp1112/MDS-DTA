"""CombinedDTA + parallel frozen ESM-2 branch (pooled architecture).

This is the original combined_dta (pooled BiLSTM protein encoder, graph drug
encoder, pooled cross-attention fusion, deep MLP head) with an additional
PARALLEL ESM-2 branch:

  * the BiLSTM protein encoder stays exactly as in the original and remains
    trainable;
  * the frozen ESM-2 branch produces one pooled vector per protein
    (masked mean+max pooling over residue features), which is merged with the
    BiLSTM pooled vector before the original fusion block;
  * ``use_esm=False`` reproduces the original model exactly.

Two ESM backends are supported:

  * ``esm_cache=<path>`` (preferred): reads the offline-precomputed features
    (from extract_esm_features.py, aligned with the dataset sample order via
    the batch-level ``esm_idx``).  No ESM inference at training time.
  * online (``esm_tag=...``, no cache): runs ESM per forward pass.  Slow.

Example (Server 1, after running extract_esm_features.py):
  python train_test.py --model combined_dta_esm --batch-size 256 \
      --model-params '{"esm_cache": "/root/autodl-tmp/mds_data/davis_esm_t33.pt"}'
"""

import torch
import torch.nn as nn

try:
    from .combined_dta import CombinedDTA
    from .combined_dta_blocks import mean_max_pool
    from .combined_dta_token import ESMProteinBranch
except ImportError:  # pragma: no cover - models dir directly on sys.path
    from combined_dta import CombinedDTA
    from combined_dta_blocks import mean_max_pool
    from combined_dta_token import ESMProteinBranch


class CombinedDTAESM(CombinedDTA):
    """Original pooled CombinedDTA with a parallel frozen ESM-2 branch."""

    def __init__(self,
                 protein_vocab=27,
                 drug_atom_feat_dim=94,
                 embed_dim=128,
                 lstm_layers=2,
                 graph_hidden=64,
                 graph_steps=3,
                 common_dim=256,
                 heads=4,
                 dropout=0.1,
                 use_esm=True,
                 esm_tag="esm2_t33_650M_UR50D",
                 esm_cache=None):
        super().__init__(
            protein_vocab=protein_vocab,
            drug_atom_feat_dim=drug_atom_feat_dim,
            embed_dim=embed_dim,
            lstm_layers=lstm_layers,
            graph_hidden=graph_hidden,
            graph_steps=graph_steps,
            common_dim=common_dim,
            heads=heads,
            dropout=dropout)

        self.use_esm = False
        self._cached_features = None
        if use_esm:
            # ESM pooled vector (mean+max of residue features) -> common_dim,
            # then merged with the BiLSTM pooled vector.
            self.esm_pool = nn.Sequential(
                nn.Linear(common_dim * 2, common_dim),
                nn.LayerNorm(common_dim),
            )
            self.merge = nn.Linear(common_dim * 2, common_dim)
            if esm_cache:
                ckpt = torch.load(esm_cache, map_location="cpu", weights_only=True)
                self._cached_features = ckpt["features"]
                self.use_esm = True
                print(
                    f"[CombinedDTAESM] ESM cache loaded from {esm_cache}: "
                    f"{tuple(self._cached_features.shape)}")
            else:
                try:
                    self.esm = ESMProteinBranch(esm_tag, common_dim)
                    self.use_esm = True
                    print(
                        f"[CombinedDTAESM] online ESM branch enabled ({esm_tag}); "
                        f"weights download on the first forward pass.")
                except Exception as exc:  # pragma: no cover
                    print(f"[CombinedDTAESM] ESM init failed, continuing "
                          f"without ESM: {exc}")

    def forward(self, data=None):
        device = next(self.parameters()).device
        data = data.to(device, non_blocking=True)

        prot_vec = self.prot_encoder(data.target.long())
        if self.use_esm:
            if self._cached_features is not None:
                esm = self._cached_features[data.esm_idx.cpu()].to(
                    device, dtype=prot_vec.dtype)
            else:
                esm = self.esm(data.target.long())
            mask = data.target.long() != 0
            esm_vec = self.esm_pool(mean_max_pool(esm, mask))
            prot_vec = self.merge(torch.cat([prot_vec, esm_vec], dim=-1))

        drug_vec = self.drug_encoder(data)

        p = prot_vec.unsqueeze(1)
        d = drug_vec.unsqueeze(1)
        p, d = self.fusion(p, d)
        return self.head(torch.cat([p.squeeze(1), d.squeeze(1)], dim=1))

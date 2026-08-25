"""Arm C: token-level encoders + masked bidirectional cross-attention.

The V2C idea (real residue/atom token interaction) rebuilt with the full set
of anti-overfitting model changes: protein-token masking, graph node dropout,
bond injection with edge perturbation, an interaction bottleneck, stochastic
depth on the fusion layers, mean+max pooling, and the residual head.

Comparing against combined_dta_aug isolates whether token-level fusion helps
once the overfitting problem is addressed.  The frozen ESM-2 branch is OFF by
default (``use_esm=False``): arm C is token fusion without ESM, and arm D
(models.combined_dta_enhanced) is this model with the ESM branch forced on.
ESM is a PARALLEL branch that receives the unmasked sequence, so the 15%
training-time masking stays a BiLSTM-only regularizer.  The ESM weights are
downloaded once through torch.hub on the first forward pass (needs network).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import EdgeConv, GatedGraphConv, GraphNorm
from torch_geometric.utils import to_dense_batch

try:
    from .combined_dta_blocks import (
        BidirectionalFusion,
        BondFeatureInjector,
        CombinedHead,
        drop_node_features,
        mask_protein_tokens,
        mean_max_pool,
    )
except ImportError:  # pragma: no cover - models dir directly on sys.path
    from combined_dta_blocks import (
        BidirectionalFusion,
        BondFeatureInjector,
        CombinedHead,
        drop_node_features,
        mask_protein_tokens,
        mean_max_pool,
    )


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


class ESMProteinBranch(nn.Module):
    """Frozen ESM-2 per-residue representations projected to ``proj_dim``.

    The ESM model is loaded lazily (first forward call) so that constructing
    the model does not require network access.  Input tokens use the same
    encoding as the rest of the project: 0 = padding, 1..20 = amino acids,
    21 = the training-time mask token (mapped to ``X`` for ESM).
    """

    def __init__(self, esm_tag, proj_dim):
        super().__init__()
        if esm_tag not in ESM_TAG_DIM:
            raise ValueError(
                f"Unknown ESM tag {esm_tag!r}; choose one of {sorted(ESM_TAG_DIM)}")
        self.tag = esm_tag
        self.repr_layer = ESM_TAG_LAYER[esm_tag]
        self.proj = nn.Linear(ESM_TAG_DIM[esm_tag], proj_dim)
        self.model = None
        self.alphabet = None
        self._loaded_device = None

    def load(self, device):
        if self.model is not None and self._loaded_device == device:
            return
        loaded = torch.hub.load("facebookresearch/esm:main", self.tag)
        if isinstance(loaded, tuple):
            # Recent esm hub entries return (model, alphabet).
            self.model, self.alphabet = loaded
        else:
            self.model = loaded
            self.alphabet = self.model.alphabet
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()
        self.model.to(device)
        self._loaded_device = device
        print(f"[CombinedDTAToken] ESM-2 {self.tag} loaded on {device}.")

    def forward(self, tokens):
        if self.model is None:
            self.load(next(self.proj.parameters()).device)

        rows = tokens.detach().cpu().tolist()
        seqs = []
        for row in rows:
            chars = []
            for token_id in row:
                if token_id == 0:
                    break  # padding is trailing in this pipeline
                chars.append(_AA_ALPHABET[token_id - 1] if 1 <= token_id <= 20 else "X")
            seqs.append(("", "".join(chars)))

        _, _, batch_tokens = self.alphabet.get_batch_converter()(seqs)
        device = next(self.proj.parameters()).device
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            out = self.model(batch_tokens, repr_layers=[self.repr_layer])
            reps = out["representations"][self.repr_layer]  # [B, L_i + 2, D]
        inner = reps[:, 1:-1]  # drop BOS / EOS

        batch_size, length = tokens.shape
        dim = inner.shape[-1]
        padded = torch.zeros(
            batch_size, length, dim, dtype=inner.dtype, device=device)
        for b in range(batch_size):
            n = int((tokens[b] != 0).sum().item())
            padded[b, :n] = inner[b, :n]
        return self.proj(padded)


class ProteinTokenEncoder(nn.Module):
    """Masked residual BiLSTM returning one token per residue."""

    def __init__(self, vocab_size, embed_dim, lstm_hidden, lstm_layers,
                 proj_dim, dropout, padding_idx, mask_prob=0.15, mask_idx=21):
        super().__init__()
        assert vocab_size > mask_idx, "vocab_size must include the mask token index"
        self.padding_idx = padding_idx
        self.mask_prob = float(mask_prob)
        self.mask_idx = int(mask_idx)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True)
        seq_dim = 2 * lstm_hidden
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_dim, seq_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(seq_dim * 4, seq_dim),
            )
            for _ in range(2)
        ])
        self.res_norms = nn.ModuleList([nn.LayerNorm(seq_dim) for _ in range(2)])
        self.proj = nn.Linear(seq_dim, proj_dim)
        self.ln = nn.LayerNorm(proj_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        if self.training:
            tokens = mask_protein_tokens(
                tokens, self.mask_prob, self.mask_idx, self.padding_idx)
        x = self.embed(tokens)
        seq, _ = self.lstm(x)
        for ffn, ln in zip(self.res_blocks, self.res_norms):
            seq = ln(seq + ffn(seq))
        mask = tokens != self.padding_idx
        seq = self.proj(seq)
        seq = self.ln(seq)
        seq = self.dropout(seq)
        return seq, mask


class GraphTokenEncoder(nn.Module):
    """Residual graph encoder returning one token per atom."""

    def __init__(self, in_dim, hidden_dim, n_steps, proj_dim, dropout,
                 node_drop_prob=0.10, num_bond_types=5, edge_noise_std=0.1):
        super().__init__()
        self.node_drop_prob = float(node_drop_prob)
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.gnorm = GraphNorm(hidden_dim)
        self.conv = GatedGraphConv(out_channels=hidden_dim, num_layers=n_steps)
        self.ln_msg = nn.LayerNorm(hidden_dim)
        self.ln_edge = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edgeconv = EdgeConv(nn=mlp)
        self.bond = BondFeatureInjector(num_bond_types, hidden_dim, edge_noise_std)
        self.proj = nn.Linear(hidden_dim, proj_dim)

    def forward(self, data):
        x0 = self.in_proj(data.x)
        if self.training:
            x0 = drop_node_features(x0, self.node_drop_prob)
        x0 = self.gnorm(x0, data.batch)

        msg = self.conv(x0, data.edge_index)
        msg = F.gelu(msg)
        msg = self.ln_msg(msg)
        msg = self.dropout(msg)
        x = x0 + msg
        x = self.bond(x, data.edge_index, getattr(data, "edge_attr", None))

        ec = self.edgeconv(x, data.edge_index)
        ec = F.gelu(ec)
        ec = self.ln_edge(ec)
        ec = self.dropout(ec)
        x = x + ec

        x = self.proj(x)
        x = self.dropout(x)
        return x, data.batch


class CombinedDTAToken(nn.Module):
    """Token-level cross-attention model with all anti-overfit changes (arm C)."""

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
                 mask_prob=0.15,
                 node_drop_prob=0.10,
                 sd_prob=0.15,
                 bottleneck_dim=128,
                 num_bond_types=5,
                 edge_noise_std=0.1,
                 use_esm=False,
                 esm_tag="esm2_t33_650M_UR50D",
                 esm_cache=None):
        super().__init__()
        self.prot_encoder = ProteinTokenEncoder(
            vocab_size=protein_vocab, embed_dim=embed_dim,
            lstm_hidden=common_dim // 2, lstm_layers=lstm_layers,
            proj_dim=common_dim, dropout=dropout, padding_idx=0,
            mask_prob=mask_prob)
        self.drug_encoder = GraphTokenEncoder(
            in_dim=drug_atom_feat_dim, hidden_dim=graph_hidden,
            n_steps=graph_steps, proj_dim=common_dim, dropout=dropout,
            node_drop_prob=node_drop_prob, num_bond_types=num_bond_types,
            edge_noise_std=edge_noise_std)

        self.fusion_in = nn.Linear(common_dim, bottleneck_dim)
        self.fusion = BidirectionalFusion(bottleneck_dim, heads, dropout, sd_prob)
        self.pool_proj = nn.Sequential(
            nn.Linear(bottleneck_dim * 2, common_dim),
            nn.LayerNorm(common_dim),
        )
        self.head = CombinedHead(common_dim * 2, common_dim, dropout)

        self.use_esm = False
        self._cached_features = None
        if use_esm and esm_cache:
            # Offline-precomputed ESM features: [N, max_len, proj_dim] fp16,
            # aligned with the dataset sample order.  Training reads rows by
            # global sample index (attached to each batch as ``esm_idx``), so
            # the frozen ESM model never runs during training.
            ckpt = torch.load(esm_cache, map_location="cpu", weights_only=True)
            self._cached_features = ckpt["features"]
            self.use_esm = True
            self.merge = nn.Linear(common_dim * 2, common_dim)
            print(
                f"[CombinedDTAToken] ESM cache loaded from {esm_cache}: "
                f"{tuple(self._cached_features.shape)}")
        elif use_esm:
            try:
                self.esm = ESMProteinBranch(esm_tag, common_dim)
                self.merge = nn.Linear(common_dim * 2, common_dim)
                self.use_esm = True
                print(
                    f"[CombinedDTAToken] ESM branch enabled ({esm_tag}); "
                    f"weights download on the first forward pass.")
            except Exception as exc:  # pragma: no cover - depends on environment
                print(f"[CombinedDTAToken] ESM init failed, continuing "
                      f"without ESM: {exc}")

    def forward(self, data=None):
        device = next(self.parameters()).device
        data = data.to(device, non_blocking=True)

        # ESM is a PARALLEL branch, not a replacement: the BiLSTM token encoder
        # always runs and stays trainable.  The 15% training-time masking is
        # applied inside ProteinTokenEncoder to its own copy of the tokens, so
        # the ESM branch below receives the *unmasked* sequence (masking is a
        # BiLSTM-only regularizer; ESM sees clean tokens at train and test).
        p, p_mask = self.prot_encoder(data.target.long())
        if self.use_esm:
            if self._cached_features is not None:
                esm = self._cached_features[data.esm_idx.cpu()].to(
                    p.device, dtype=p.dtype)
            else:
                esm = self.esm(data.target.long())
            p = self.merge(torch.cat([p, esm], dim=-1))
        x, batch = self.drug_encoder(data)
        d, d_mask = to_dense_batch(x, batch)

        p = self.fusion_in(p)
        d = self.fusion_in(d)
        p, d = self.fusion(p, d, p_mask, d_mask)

        gp = self.pool_proj(mean_max_pool(p, p_mask))
        gd = self.pool_proj(mean_max_pool(d, d_mask))
        return self.head(torch.cat([gp, gd], dim=1))

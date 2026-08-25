"""Shared building blocks for the CombinedDTA enhanced / ablation models.

All augmentation in this module is model-internal: it is applied inside
``forward()`` and only when ``self.training`` is True, so it has zero effect
at inference time.  This matches the design principle that data augmentation
for these models belongs to the model (like ``nn.Dropout``), not to the
training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


def mask_protein_tokens(tokens, p, mask_idx, pad_idx=0):
    """Randomly replace ``p`` of the non-padding tokens with ``mask_idx``.

    The masked position is *kept* (it still participates in pooling) but its
    identity is hidden behind the learnable mask embedding.  Padding tokens
    are never masked.
    """
    if p <= 0.0:
        return tokens
    mask = (torch.rand_like(tokens.float()) < p) & (tokens != pad_idx)
    return torch.where(mask, torch.full_like(tokens, mask_idx), tokens)


def drop_node_features(x, p):
    """Zero out ``p`` of the node feature vectors (node dropout, training only).

    The graph topology is kept; dropped nodes simply contribute nothing to the
    subsequent layers.  This is a cheap, effective regularizer for graph
    encoders.
    """
    if p <= 0.0:
        return x
    drop = (torch.rand(x.shape[0], 1, device=x.device) < p).float()
    return x * (1.0 - drop)


class BondFeatureInjector(nn.Module):
    """Optionally inject bond-type information into node representations.

    ``edge_attr`` is expected to be one integer bond type per directed edge
    (0 single, 1 double, 2 triple, 3 aromatic, 4 other) or a one-hot vector
    per edge.  When ``edge_attr`` is missing (legacy processed data) the
    module is a no-op, so the same model code runs on old and new data.

    During training a small Gaussian perturbation is added to the bond
    embedding (edge perturbation).
    """

    def __init__(self, num_bond_types, hidden_dim, edge_noise_std=0.0):
        super().__init__()
        self.embed = nn.Embedding(num_bond_types, hidden_dim)
        self.edge_noise_std = float(edge_noise_std)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is None:
            return x
        if edge_attr.dim() > 1 and edge_attr.size(-1) > 1:
            ea = edge_attr.argmax(dim=-1).long()
        else:
            ea = edge_attr.long()
            if ea.dim() > 1:
                ea = ea.squeeze(-1)
        ea = ea.clamp(min=0, max=self.embed.num_embeddings - 1)
        emb = self.embed(ea)
        if self.training and self.edge_noise_std > 0.0:
            emb = emb + torch.randn_like(emb) * self.edge_noise_std
        msg = scatter(emb, edge_index[0], dim=0, dim_size=x.shape[0], reduce="mean")
        return x + msg


class TokenFusionLayer(nn.Module):
    """One layer of masked bidirectional cross-attention with gated residual.

    Mirrors the residual/gate/FFN skeleton of the original fusion block, but
    the softmax runs over the full token set of the other modality.
    """

    def __init__(self, embed_dim, heads, dropout):
        super().__init__()

        def _ffn():
            return nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
            )

        self.p2d = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.d2p = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.ln1_p = nn.LayerNorm(embed_dim)
        self.ln1_d = nn.LayerNorm(embed_dim)
        self.ffn_p = _ffn()
        self.ffn_d = _ffn()
        self.lnffn_p = nn.LayerNorm(embed_dim)
        self.lnffn_d = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate_p = nn.Parameter(torch.tensor(0.5))
        self.gate_d = nn.Parameter(torch.tensor(0.5))

    def forward(self, p, d, p_mask, d_mask):
        p_att, _ = self.p2d(p, d, d, key_padding_mask=~d_mask, need_weights=False)
        p = self.ln1_p(p + self.dropout(self.gate_p * p_att))
        p = self.lnffn_p(p + self.dropout(self.ffn_p(p)))

        d_att, _ = self.d2p(d, p, p, key_padding_mask=~p_mask, need_weights=False)
        d = self.ln1_d(d + self.dropout(self.gate_d * d_att))
        d = self.lnffn_d(d + self.dropout(self.ffn_d(d)))
        return p, d


class BidirectionalFusion(nn.Module):
    """Two layers of masked bidirectional cross-attention with stochastic depth.

    During training each layer is skipped with probability ``sd_prob``
    (stochastic depth).  At inference both layers always run.
    """

    def __init__(self, embed_dim, heads, dropout, sd_prob=0.15):
        super().__init__()
        self.layer1 = TokenFusionLayer(embed_dim, heads, dropout)
        self.layer2 = TokenFusionLayer(embed_dim, heads, dropout)
        self.sd_prob = float(sd_prob)

    def forward(self, p, d, p_mask, d_mask):
        p, d = self._maybe(self.layer1, p, d, p_mask, d_mask)
        p, d = self._maybe(self.layer2, p, d, p_mask, d_mask)
        return p, d

    def _maybe(self, layer, p, d, p_mask, d_mask):
        if self.training and self.sd_prob > 0.0 and torch.rand(1).item() < self.sd_prob:
            return p, d
        return layer(p, d, p_mask, d_mask)


def mean_max_pool(tokens, mask):
    """Masked mean + max pooling over the token dimension."""
    m = mask.float().unsqueeze(-1)
    denom = m.sum(dim=1).clamp(min=1.0)
    mean_pool = (tokens * m).sum(dim=1) / denom
    # -inf (not -1e9) is used because the tokens can be fp16 under AMP:
    # -1e9 overflows fp16, -inf is natively representable.
    seq_masked = tokens.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    max_pool = seq_masked.amax(dim=1)
    has_tokens = mask.any(dim=1).unsqueeze(1)
    max_pool = torch.where(has_tokens, max_pool, torch.zeros_like(max_pool))
    return torch.cat([mean_pool, max_pool], dim=1)


class CombinedHead(nn.Module):
    """Deep MLP head with a global residual.

    out = head(x) + skip(x)

    ``skip`` is zero-initialised so training starts exactly like a plain head
    and the residual path grows only if it helps.  There is deliberately no
    sigmoid output bound: label ranges differ across datasets (Davis pKd,
    KIBA, BindingDB), so a fixed bound would need per-dataset tuning and it
    interacted badly with the floor loss.
    """

    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.skip = nn.Linear(in_dim, 1)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

    def forward(self, x):
        return self.head(x) + self.skip(x)

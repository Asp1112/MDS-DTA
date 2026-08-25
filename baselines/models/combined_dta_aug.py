"""Arm B: original pooled architecture + all model-internal changes.

Keeps the original pooled BiLSTM / graph encoders and the two-layer fusion
skeleton, and adds the model-side changes under test:

  * 15% protein-token masking and 10% graph node dropout (training only);
  * bond-type feature injection with Gaussian edge perturbation
    (no-op when ``edge_attr`` is absent from the processed data);
  * interaction bottleneck (common_dim -> bottleneck_dim) before fusion;
  * stochastic depth on the two fusion layers;
  * prediction head with a global residual and a bounded output.

Comparing against combined_dta_control isolates the model-side changes while
holding the (pooled) fusion paradigm fixed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import EdgeConv, GatedGraphConv, GraphNorm, global_mean_pool


try:
    from .combined_dta_blocks import (
        BidirectionalFusion,
        BondFeatureInjector,
        CombinedHead,
        drop_node_features,
        mask_protein_tokens,
    )
except ImportError:  # pragma: no cover - models dir directly on sys.path
    from combined_dta_blocks import (
        BidirectionalFusion,
        BondFeatureInjector,
        CombinedHead,
        drop_node_features,
        mask_protein_tokens,
    )


class ProteinBiLSTMEncoder(nn.Module):
    """Masked residual BiLSTM encoder with mean+max pooling and token masking."""

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
        self.proj = nn.Linear(seq_dim * 2, proj_dim)
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

        pad_mask = (tokens != self.padding_idx)
        pm = pad_mask.unsqueeze(-1)
        m = pm.float()
        denom = m.sum(dim=1).clamp(min=1.0)
        mean_pool = (seq * m).sum(dim=1) / denom
        seq_masked = seq.masked_fill(~pm, -1e9)
        max_pool = seq_masked.amax(dim=1)
        has_tokens = pad_mask.any(dim=1).unsqueeze(1)
        max_pool = torch.where(has_tokens, max_pool, torch.zeros_like(max_pool))

        feat = torch.cat([mean_pool, max_pool], dim=1)
        feat = self.proj(feat)
        feat = self.ln(feat)
        feat = self.dropout(feat)
        return feat


class GraphEncoder(nn.Module):
    """Residual graph encoder with node dropout and optional bond features."""

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
        self.pool = global_mean_pool
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

        x = self.pool(x, data.batch)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class CombinedDTAAug(nn.Module):
    """Original pooled architecture + all model-internal changes (arm B)."""

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
                 edge_noise_std=0.1):
        super().__init__()
        self.prot_encoder = ProteinBiLSTMEncoder(
            vocab_size=protein_vocab, embed_dim=embed_dim,
            lstm_hidden=common_dim // 2, lstm_layers=lstm_layers,
            proj_dim=common_dim, dropout=dropout, padding_idx=0,
            mask_prob=mask_prob)
        self.drug_encoder = GraphEncoder(
            in_dim=drug_atom_feat_dim, hidden_dim=graph_hidden,
            n_steps=graph_steps, proj_dim=common_dim, dropout=dropout,
            node_drop_prob=node_drop_prob, num_bond_types=num_bond_types,
            edge_noise_std=edge_noise_std)

        self.fusion_in = nn.Linear(common_dim, bottleneck_dim)
        self.fusion = BidirectionalFusion(bottleneck_dim, heads, dropout, sd_prob)
        self.fusion_out = nn.Linear(bottleneck_dim, common_dim)
        self.head = CombinedHead(common_dim * 2, common_dim, dropout)

    def forward(self, data=None):
        device = next(self.parameters()).device
        data = data.to(device, non_blocking=True)

        prot_vec = self.prot_encoder(data.target.long())
        drug_vec = self.drug_encoder(data)

        batch = prot_vec.shape[0]
        p = prot_vec.unsqueeze(1)
        d = drug_vec.unsqueeze(1)
        p_mask = torch.ones(batch, 1, dtype=torch.bool, device=p.device)
        d_mask = torch.ones(batch, 1, dtype=torch.bool, device=p.device)

        p = self.fusion_in(p)
        d = self.fusion_in(d)
        p, d = self.fusion(p, d, p_mask, d_mask)
        p = self.fusion_out(p).squeeze(1)
        d = self.fusion_out(d).squeeze(1)
        return self.head(torch.cat([p, d], dim=1))

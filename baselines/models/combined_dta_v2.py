"""CombinedDTA V2 (token-level cross-attention with learned-query pooling).

Built from the original combined_dta.py (no LSTM inter-layer dropout).
Revision addressing the reviewer's concern that pooling to a single token
before cross-attention degenerates the softmax (one key/value token).

Changes vs. the original CombinedDTA:
  1. The protein and ligand encoders now emit token-level representations
     (one token per residue / per atom) instead of a single pooled vector.
  2. Cross-attention is applied between the two full token sets with
     padding masks, so the softmax selects among many keys (real attention).
  3. After fusion, each modality is summarised by a learned-query attention
     pooling module, replacing the mean+max global pooling before fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GatedGraphConv, GraphNorm, EdgeConv
from torch_geometric.utils import to_dense_batch


class ProteinTokenEncoder(nn.Module):
    """Masked residual BiLSTM encoder returning one token per residue."""

    def __init__(self, vocab_size, embed_dim, lstm_hidden, lstm_layers,
                 proj_dim, dropout, padding_idx):
        super().__init__()
        self.padding_idx = padding_idx
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
        x = self.embed(tokens)
        seq, _ = self.lstm(x)
        for ffn, ln in zip(self.res_blocks, self.res_norms):
            seq = ln(seq + ffn(seq))
        mask = tokens != self.padding_idx  # [B, L]
        seq = self.proj(seq)
        seq = self.ln(seq)
        seq = self.dropout(seq)
        return seq, mask


class GraphTokenEncoder(nn.Module):
    """Sequential residual graph encoder returning one token per atom."""

    def __init__(self, in_dim, hidden_dim, n_steps, proj_dim, dropout):
        super().__init__()
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
        self.proj = nn.Linear(hidden_dim, proj_dim)

    def forward(self, data):
        x0 = self.in_proj(data.x)
        x0 = self.gnorm(x0, data.batch)
        msg = self.conv(x0, data.edge_index)
        msg = F.gelu(msg)
        msg = self.ln_msg(msg)
        msg = self.dropout(msg)
        x = x0 + msg
        ec = self.edgeconv(x, data.edge_index)
        ec = F.gelu(ec)
        ec = self.ln_edge(ec)
        ec = self.dropout(ec)
        x = x + ec
        x = self.proj(x)
        x = self.dropout(x)
        return x, data.batch


def masked_mha(mha, query, key, value, key_mask):
    """Multihead attention with a boolean validity mask (True = valid token)."""
    return mha(query, key, value,
               key_padding_mask=~key_mask, need_weights=False)[0]


class TokenCrossAttentionFusion(nn.Module):
    """Two layers of masked bidirectional token-level cross-attention.

    Keeps the residual/gate/FFN skeleton of the original fusion block, but
    the softmax now runs over the full token sets of the other modality.
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

        self.p2d_1 = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.d2p_1 = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.ln1_p = nn.LayerNorm(embed_dim)
        self.ln1_d = nn.LayerNorm(embed_dim)
        self.ffn_p_1 = _ffn()
        self.ffn_d_1 = _ffn()
        self.lnffn1_p = nn.LayerNorm(embed_dim)
        self.lnffn1_d = nn.LayerNorm(embed_dim)

        self.p2d_2 = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.d2p_2 = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.ln2_p = nn.LayerNorm(embed_dim)
        self.ln2_d = nn.LayerNorm(embed_dim)
        self.ffn_p_2 = _ffn()
        self.ffn_d_2 = _ffn()
        self.lnffn2_p = nn.LayerNorm(embed_dim)
        self.lnffn2_d = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.gate_p1 = nn.Parameter(torch.tensor(0.5))
        self.gate_d1 = nn.Parameter(torch.tensor(0.5))
        self.gate_p2 = nn.Parameter(torch.tensor(0.5))
        self.gate_d2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, p, p_mask, d, d_mask):
        # Layer 1
        p = self.ln1_p(p + self.dropout(self.gate_p1 * masked_mha(self.p2d_1, p, d, d, d_mask)))
        p = self.lnffn1_p(p + self.dropout(self.ffn_p_1(p)))
        d = self.ln1_d(d + self.dropout(self.gate_d1 * masked_mha(self.d2p_1, d, p, p, p_mask)))
        d = self.lnffn1_d(d + self.dropout(self.ffn_d_1(d)))
        # Layer 2
        p = self.ln2_p(p + self.dropout(self.gate_p2 * masked_mha(self.p2d_2, p, d, d, d_mask)))
        p = self.lnffn2_p(p + self.dropout(self.ffn_p_2(p)))
        d = self.ln2_d(d + self.dropout(self.gate_d2 * masked_mha(self.d2p_2, d, p, p, p_mask)))
        d = self.lnffn2_d(d + self.dropout(self.ffn_d_2(d)))
        return p, d


class AttentionPool(nn.Module):
    """Learned-query attention pooling: one query token attends over a set."""

    def __init__(self, embed_dim, heads, dropout):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.mha = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, tokens, mask):
        query = self.query.expand(tokens.shape[0], 1, tokens.shape[2])
        out = masked_mha(self.mha, query, tokens, tokens, mask)  # [B, 1, D]
        return self.ln(out.squeeze(1))


class CombinedDTAV2(nn.Module):
    def __init__(self,
                 protein_vocab=27,
                 drug_atom_feat_dim=94,
                 embed_dim=128,
                 lstm_layers=2,
                 graph_hidden=64,
                 graph_steps=3,
                 common_dim=256,
                 heads=4,
                 dropout=0.1):
        super().__init__()
        self.prot_encoder = ProteinTokenEncoder(
            vocab_size=protein_vocab, embed_dim=embed_dim,
            lstm_hidden=common_dim // 2, lstm_layers=lstm_layers,
            proj_dim=common_dim, dropout=dropout, padding_idx=0)
        self.drug_encoder = GraphTokenEncoder(
            in_dim=drug_atom_feat_dim, hidden_dim=graph_hidden,
            n_steps=graph_steps, proj_dim=common_dim, dropout=dropout)
        self.fusion = TokenCrossAttentionFusion(embed_dim=common_dim, heads=heads, dropout=dropout)
        self.pool_p = AttentionPool(common_dim, heads, dropout)
        self.pool_d = AttentionPool(common_dim, heads, dropout)
        self.head = self._build_head(common_dim, dropout)

    @staticmethod
    def _build_head(d, dropout):
        return nn.Sequential(
            nn.Linear(d * 2, d * 2),
            nn.GELU(),
            nn.LayerNorm(d * 2),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, 1),
        )

    def forward(self, data=None):
        device = next(self.parameters()).device
        data = data.to(device, non_blocking=True)
        p, p_mask = self.prot_encoder(data.target.long())
        x, batch = self.drug_encoder(data)
        d, d_mask = to_dense_batch(x, batch)  # [B, N_max, D], [B, N_max]
        p, d = self.fusion(p, p_mask, d, d_mask)
        gp = self.pool_p(p, p_mask)
        gd = self.pool_d(d, d_mask)
        return self.head(torch.cat([gp, gd], dim=1))

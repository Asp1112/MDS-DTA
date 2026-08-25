"""CombinedDTA V2-B (ablation: gated MLP mixing, no cross-attention).

Built from the original combined_dta.py (no LSTM inter-layer dropout).
Same encoders and prediction head as the original model, but the fusion
block replaces every MultiheadAttention with a single learned linear
projection of the other modality's pooled vector. There is no attention
softmax anywhere, so this variant isolates the contribution of the
attention component itself while keeping the residual gate + FFN skeleton.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool, GatedGraphConv, GraphNorm, EdgeConv


class ProteinBiLSTMEncoder(nn.Module):
    """Masked residual BiLSTM encoder with mean+max global pooling."""

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
        self.proj = nn.Linear(seq_dim * 2, proj_dim)
        self.ln = nn.LayerNorm(proj_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        x = self.embed(tokens)
        seq, _ = self.lstm(x)
        for ffn, ln in zip(self.res_blocks, self.res_norms):
            seq = ln(seq + ffn(seq))

        pad_mask = tokens != self.padding_idx
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
    """Sequential residual graph encoder with global mean pooling."""

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
        self.pool = global_mean_pool
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
        x = self.pool(x, data.batch)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class GatedMLPFusion(nn.Module):
    """Two layers of gated cross-modal linear mixing (no attention)."""

    def __init__(self, embed_dim, dropout):
        super().__init__()

        def _ffn():
            return nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
            )

        self.p2d_1 = nn.Linear(embed_dim, embed_dim)
        self.d2p_1 = nn.Linear(embed_dim, embed_dim)
        self.ln1_p = nn.LayerNorm(embed_dim)
        self.ln1_d = nn.LayerNorm(embed_dim)
        self.ffn_p_1 = _ffn()
        self.ffn_d_1 = _ffn()
        self.lnffn1_p = nn.LayerNorm(embed_dim)
        self.lnffn1_d = nn.LayerNorm(embed_dim)

        self.p2d_2 = nn.Linear(embed_dim, embed_dim)
        self.d2p_2 = nn.Linear(embed_dim, embed_dim)
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

    def forward(self, p, d):
        p = self.ln1_p(p + self.dropout(self.gate_p1 * self.p2d_1(d)))
        p = self.lnffn1_p(p + self.dropout(self.ffn_p_1(p)))
        d = self.ln1_d(d + self.dropout(self.gate_d1 * self.d2p_1(p)))
        d = self.lnffn1_d(d + self.dropout(self.ffn_d_1(d)))
        p = self.ln2_p(p + self.dropout(self.gate_p2 * self.p2d_2(d)))
        p = self.lnffn2_p(p + self.dropout(self.ffn_p_2(p)))
        d = self.ln2_d(d + self.dropout(self.gate_d2 * self.d2p_2(p)))
        d = self.lnffn2_d(d + self.dropout(self.ffn_d_2(d)))
        return p, d


class CombinedDTAV2B(nn.Module):
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
        self.prot_encoder = ProteinBiLSTMEncoder(
            vocab_size=protein_vocab, embed_dim=embed_dim,
            lstm_hidden=common_dim // 2, lstm_layers=lstm_layers,
            proj_dim=common_dim, dropout=dropout, padding_idx=0)
        self.drug_encoder = GraphEncoder(
            in_dim=drug_atom_feat_dim, hidden_dim=graph_hidden,
            n_steps=graph_steps, proj_dim=common_dim, dropout=dropout)
        self.fusion = GatedMLPFusion(embed_dim=common_dim, dropout=dropout)
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
        prot_vec = self.prot_encoder(data.target.long())
        drug_vec = self.drug_encoder(data)
        prot_vec, drug_vec = self.fusion(prot_vec, drug_vec)
        return self.head(torch.cat([prot_vec, drug_vec], dim=1))

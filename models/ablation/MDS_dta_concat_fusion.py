import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GatedGraphConv, GraphNorm, EdgeConv


class ProteinBiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_hidden, lstm_layers,
                 proj_dim, dropout, padding_idx):
        super().__init__()
        self.padding_idx = padding_idx
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, lstm_layers, batch_first=True, bidirectional=True)
        seq_dim = 2 * lstm_hidden
        self.res_blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(seq_dim, seq_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(seq_dim * 4, seq_dim))
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
        pad_mask = (tokens != self.padding_idx)
        m = pad_mask.float().unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        mean_pool = (seq * m).sum(dim=1) / denom
        seq_masked = seq.masked_fill(~pad_mask.unsqueeze(-1), float('-inf'))
        max_pool = seq_masked.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        feat = torch.cat([mean_pool, max_pool], dim=1)
        feat = self.proj(feat)
        feat = self.ln(feat)
        feat = self.dropout(feat)
        return feat


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_steps, proj_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.gnorm = GraphNorm(hidden_dim)
        self.conv = GatedGraphConv(out_channels=hidden_dim, num_layers=n_steps)
        self.ln_msg = nn.LayerNorm(hidden_dim)
        self.ln_edge = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
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


class MDSDTAConcatFusion(nn.Module):
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
        self.prot_encoder = ProteinBiLSTMEncoder(vocab_size=protein_vocab,
                                                 embed_dim=embed_dim,
                                                 lstm_hidden=common_dim // 2,
                                                 lstm_layers=lstm_layers,
                                                 proj_dim=common_dim,
                                                 dropout=dropout,
                                                 padding_idx=0)
        self.drug_encoder = GraphEncoder(in_dim=drug_atom_feat_dim, hidden_dim=graph_hidden,
                                         n_steps=graph_steps, proj_dim=common_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim * 2), nn.GELU(), nn.LayerNorm(common_dim * 2), nn.Dropout(dropout),
            nn.Linear(common_dim * 2, common_dim), nn.GELU(), nn.LayerNorm(common_dim), nn.Dropout(dropout),
            nn.Linear(common_dim, common_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(common_dim, common_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(common_dim // 2, 1)
        )
    def forward(self, data):
        device = next(self.parameters()).device
        data = data.to(device, non_blocking=True)
        protein_seq = data.target.long().to(device)
        prot_vec = self.prot_encoder(protein_seq)
        drug_vec = self.drug_encoder(data)
        x = torch.cat([prot_vec, drug_vec], dim=1)
        out = self.head(x)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GatedGraphConv, GraphNorm, EdgeConv
from torch_geometric.utils import to_dense_batch


class ProteinBiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_hidden, lstm_layers, proj_dim, dropout, padding_idx):
        super().__init__()
        self.padding_idx = padding_idx
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True)
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


class Compound1DCNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, proj_dim, dropout):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=proj_dim, kernel_size=3, padding=1)
        self.act = nn.ReLU()
    def forward(self, data):
        x_dense, _ = to_dense_batch(data.x, data.batch)
        x = x_dense.transpose(1, 2)
        x = self.act(self.conv(x))
        feat = x.mean(dim=2)
        return feat


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, heads, dropout):
        super().__init__()
        self.prot_to_drug_1 = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.drug_to_prot_1 = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.ln1_p = nn.LayerNorm(embed_dim)
        self.ln1_d = nn.LayerNorm(embed_dim)
        self.ffn_p_1 = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim))
        self.ffn_d_1 = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim))
        self.lnffn1_p = nn.LayerNorm(embed_dim)
        self.lnffn1_d = nn.LayerNorm(embed_dim)
        self.prot_to_drug_2 = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.drug_to_prot_2 = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.ln2_p = nn.LayerNorm(embed_dim)
        self.ln2_d = nn.LayerNorm(embed_dim)
        self.ffn_p_2 = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim))
        self.ffn_d_2 = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim))
        self.lnffn2_p = nn.LayerNorm(embed_dim)
        self.lnffn2_d = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate_p1 = nn.Parameter(torch.tensor(0.5))
        self.gate_d1 = nn.Parameter(torch.tensor(0.5))
        self.gate_p2 = nn.Parameter(torch.tensor(0.5))
        self.gate_d2 = nn.Parameter(torch.tensor(0.5))
    def forward(self, prot_seq, drug_seq):
        p_att1, _ = self.prot_to_drug_1(query=prot_seq, key=drug_seq, value=drug_seq)
        prot_seq = self.ln1_p(prot_seq + self.dropout(self.gate_p1 * p_att1))
        prot_seq = self.lnffn1_p(prot_seq + self.dropout(self.ffn_p_1(prot_seq)))
        d_att1, _ = self.drug_to_prot_1(query=drug_seq, key=prot_seq, value=prot_seq)
        drug_seq = self.ln1_d(drug_seq + self.dropout(self.gate_d1 * d_att1))
        drug_seq = self.lnffn1_d(drug_seq + self.dropout(self.ffn_d_1(drug_seq)))
        p_att2, _ = self.prot_to_drug_2(query=prot_seq, key=drug_seq, value=drug_seq)
        prot_seq = self.ln2_p(prot_seq + self.dropout(self.gate_p2 * p_att2))
        prot_seq = self.lnffn2_p(prot_seq + self.dropout(self.ffn_p_2(prot_seq)))
        d_att2, _ = self.drug_to_prot_2(query=drug_seq, key=prot_seq, value=prot_seq)
        drug_seq = self.ln2_d(drug_seq + self.dropout(self.gate_d2 * d_att2))
        drug_seq = self.lnffn2_d(drug_seq + self.dropout(self.ffn_d_2(drug_seq)))
        return prot_seq, drug_seq


class MDSDTA_compound(nn.Module):
    def __init__(self,
                 protein_vocab=27,
                 drug_atom_feat_dim=94,
                 embed_dim=128,
                 lstm_layers=1,
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
        self.drug_encoder = Compound1DCNNEncoder(in_dim=drug_atom_feat_dim,
                                                 hidden_dim=graph_hidden,
                                                 proj_dim=common_dim,
                                                 dropout=dropout)
        self.fusion = CrossAttentionFusion(embed_dim=common_dim, heads=heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim * 2),
            nn.GELU(),
            nn.LayerNorm(common_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(common_dim * 2, common_dim),
            nn.GELU(),
            nn.LayerNorm(common_dim),
            nn.Dropout(dropout),
            nn.Linear(common_dim, common_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim, common_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim // 2, 1)
        )
    def forward(self, data=None):
        device = next(self.parameters()).device
        data = data.to(device, non_blocking=True)
        prot_tokens = data.target
        protein_seq = prot_tokens.long().to(device)
        drug_graph = data
        prot_vec = self.prot_encoder(protein_seq)
        prot_seq = prot_vec.unsqueeze(1)
        drug_vec = self.drug_encoder(drug_graph)
        drug_seq = drug_vec.unsqueeze(1)
        prot_after, drug_after = self.fusion(prot_seq, drug_seq)
        prot_pooled = prot_after.mean(dim=1)
        drug_pooled = drug_after.mean(dim=1)
        x = torch.cat([prot_pooled, drug_pooled], dim=1)
        out = self.head(x)
        return out

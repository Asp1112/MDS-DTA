"""Arm D: token-level model + frozen ESM-2 branch (always on).

``CombinedDTAEnhanced`` is exactly ``CombinedDTAToken`` with ``use_esm=True``
forced, so arm D differs from arm C (token, ESM off) only by the frozen ESM
branch.  ESM is a PARALLEL branch: the BiLSTM keeps running and stays
trainable, and the 15% training-time masking applies only to the BiLSTM tokens
(the ESM branch receives the unmasked sequence).  The ESM weights download
once via torch.hub on the first forward pass, so the first run needs network
access.
"""

try:
    from .combined_dta_token import CombinedDTAToken
except ImportError:  # pragma: no cover - models dir directly on sys.path
    from combined_dta_token import CombinedDTAToken


class CombinedDTAEnhanced(CombinedDTAToken):
    """Token-level cross-attention with the frozen ESM-2 branch forced on."""

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
                 esm_tag="esm2_t33_650M_UR50D",
                 esm_cache=None):
        # use_esm is deliberately not exposed: arm D always has ESM on.  The
        # explicit signature (instead of *args/**kwargs) lets train_test.py's
        # model-params filter pass esm_tag (and any other model knob) through.
        super().__init__(
            protein_vocab=protein_vocab,
            drug_atom_feat_dim=drug_atom_feat_dim,
            embed_dim=embed_dim,
            lstm_layers=lstm_layers,
            graph_hidden=graph_hidden,
            graph_steps=graph_steps,
            common_dim=common_dim,
            heads=heads,
            dropout=dropout,
            mask_prob=mask_prob,
            node_drop_prob=node_drop_prob,
            sd_prob=sd_prob,
            bottleneck_dim=bottleneck_dim,
            num_bond_types=num_bond_types,
            edge_noise_std=edge_noise_std,
            use_esm=True,
            esm_tag=esm_tag,
            esm_cache=esm_cache)

# Ablation models

Six architectural ablation variants of the MDS model (manuscript
Supplementary Table 11). Each file defines an `nn.Module` class
(`MDSDTA_*`) with the same input/output interface as the main model:

* `MDS_dta_compound_1dcnn.py` — molecular graph encoder replaced by a 1D CNN.
* `MDS_dta_protein_1dcnn.py` — protein encoder replaced by a 1D CNN.
* `MDS_dta_both_1dcnn.py` — both unimodal encoders replaced by 1D CNNs.
* `MDS_dta_concat_fusion.py` — gated cross-modal fusion replaced by
  concatenation.
* `MDS_dta_combined_fc.py` — MLP regression head replaced by a fully
  connected layer.
* `MDS_dta_protein_transformer.py` — BiLSTM protein encoder replaced by a
  Transformer encoder.

These variants are invoked through the training harness, e.g.
`python train_test.py --dataset davis --model MDS_dta_both_1dcnn --test-fold 0`.

# Randomization experiments

Sample-level six-fold control: shuffle the protein input (`x1`), the compound
graph (`x2`) or the labels (`y`) while keeping everything else unchanged.

```bash
python prepare_randomization.py --datasets davis kiba bindingdb
python train_random.py --dataset davis --mode x2 --fold 0 --model MDS_dta
bash run_randomization.sh --model MDS_dta --datasets davis kiba bindingdb
```

Prepared datasets and splits are in `data/`.

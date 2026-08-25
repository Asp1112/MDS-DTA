# Cold-start experiments

Entity-level six-fold evaluation: unique compounds and/or targets are split
into six disjoint groups; fold N uses group N as test, group N+1 as
validation and the remaining groups for training.

```bash
python prepare_cold_start.py --datasets davis kiba bindingdb
python train_cold.py --dataset davis --setting cold_drug --fold 0 --model MDS_dta
bash run_cold_start.sh --model MDS_dta --datasets davis kiba bindingdb
```

Prepared datasets and splits are in `data/`.

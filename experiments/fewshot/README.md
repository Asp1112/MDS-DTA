# Reduced-data (few-shot) experiments

Sample-level six-fold evaluation with 50%, 25% and 10% of the training data
to test data-scarcity behavior.

```bash
python prepare_fewshot.py --datasets davis kiba bindingdb
python train_fewshot.py --dataset davis --setting fs25 --fold 0 --model MDS_dta
bash run_fewshot.sh --model MDS_dta --datasets davis kiba bindingdb
```

Prepared datasets and splits are in `data/`.

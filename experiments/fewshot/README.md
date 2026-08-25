# 少样本实验（六折划分）

## 协议

少样本实验沿用样本级六折轮转协议：第 N 折测试集 = 第 N 折样本，验证集 =
第 (N+1) mod 6 折样本，完整训练池 = 其余 4 折。随后按 50% / 25% / 10% 从该
训练池中无放回抽样（固定种子 42），得到 `fs50` / `fs25` / `fs10` 三个设置。
验证集与测试集始终保持完整，只减少训练数据量。

## 1. 生成数据集

```bash
cd experiments/fewshot
python prepare_fewshot.py --datasets davis kiba bindingdb
```

输出：

- `data/splits/<dataset>/fs50/fold_0..5.json`
- `data/splits/<dataset>/fs25/fold_0..5.json`
- `data/splits/<dataset>/fs10/fold_0..5.json`

每个清单包含抽样后的 `train_indices`、完整验证/测试索引、训练池大小与
抽样比例审计。

## 2. 训练

```bash
cd experiments/fewshot

# 单个折（示例）
python train_fewshot.py \
  --dataset davis --setting fs25 --fold 0 --model combined_dta_edge

# davis + kiba + bindingdb、fs50/fs25/fs10、全部 6 折
bash run_fewshot.sh --datasets davis kiba bindingdb
```

## 常用参数

- `--dry`：2 个 epoch 冒烟测试。
- `--results-root <路径>`：默认 `experiments/results/fewshot`。
- `--skip-done`：断点续跑时跳过已完成运行。
- `--model`：默认 `combined_dta`，可换任意 `models/` 下的 MDS 模型。

## 输出

与冷启动相同的结果布局（config / split_indices / history / top3 检查点 /
validation_summary / test_metrics / test_predictions.npz）。

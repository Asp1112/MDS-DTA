# 冷启动实验（六折划分）

## 协议

冷启动采用**实体级六折**，与论文的六折划分要求一致：

- `cold_drug`（化合物冷启动/单冷）：把全部唯一化合物分成 6 个互不重叠的组。第 N 折把第 N 组化合物的全部相互作用作为测试集，第 (N+1) mod 6 组作为验证集，其余 4 组作为训练集。测试化合物在训练中完全不可见。
- `cold_target`（靶点冷启动/单冷）：对唯一靶点做同样的 6 组划分。
- `cold_both`（双冷）：化合物和靶点各分成 6 组。第 N 折的测试集 =「化合物组 N 且靶点组 N」的全部相互作用；验证集 =「组 N+1 且组 N+1」；训练集 = 化合物和靶点都不属于 N、N+1 组的样本。测试药物与测试靶点在训练集中均不可见。

每个设置、每个折生成一个 `fold_<N>.json` 清单，记录
`train_indices` / `validation_indices` / `test_indices`（都是
`<dataset>_sixfold_all.csv` 的样本序号）、实体分组与不重叠审计。

## 1. 生成数据集

```bash
cd experiments/cold_start
python prepare_cold_start.py --datasets davis kiba bindingdb
```

输出：

- `data/splits/<dataset>/cold_drug/fold_0..5.json`
- `data/splits/<dataset>/cold_target/fold_0..5.json`
- `data/splits/<dataset>/cold_both/fold_0..5.json`
- `data/<dataset>/summary.json`（各折样本数与不重叠审计）

## 2. DeepDTA 运行指令

```bash
cd experiments/cold_start
# 单个折（示例）
python train_cold.py \
  --dataset davis --setting cold_drug --fold 0 --model deepdta

# davis + kiba + bindingdb 全部设置、全部 6 折
bash run_cold_start.sh --model deepdta --datasets davis kiba bindingdb
```

## 3. GraphDTA 运行指令

支持 `graphdta_gcn` / `graphdta_gat` / `graphdta_gat_gcn` / `graphdta_ginconv`
四个变体：

```bash
cd experiments/cold_start
# 单个折（示例）
python train_cold.py \
  --dataset davis --setting cold_target --fold 0 --model graphdta_gcn

# 全部数据集、全部 6 折（GCN 变体）
bash run_cold_start.sh --model graphdta_gcn --datasets davis kiba bindingdb
# 其他变体把 --model 换成 graphdta_gat / graphdta_gat_gcn / graphdta_ginconv
```

## 4. DeepDTAGen 运行指令

```bash
cd experiments/cold_start
# 单个折（示例）
python train_cold.py \
  --dataset davis --setting cold_both --fold 0 --model deepdtagen

# 全部数据集、全部 6 折
bash run_cold_start.sh --model deepdtagen --datasets davis kiba bindingdb
```

## 5. MDS 模型族运行指令

```bash
cd experiments/cold_start
python train_cold.py \
  --dataset davis --setting cold_drug --fold 0 --model combined_dta_edge
bash run_cold_start.sh --model combined_dta_edge --datasets davis kiba bindingdb
# 其他 MDS 模型：combined_dta / combined_dta_lstmdrop / combined_dta_token ...
```

## 常用参数

- `--dry`：2 个 epoch 的冒烟测试（验证代码与数据可跑通）。
- `--results-root <路径>`：结果输出目录（默认
  `experiments/results/cold_start`，即大容量数据盘）。
- `--skip-done`：已有完整 `test_metrics.json` 的运行自动跳过，方便断点续跑。
- `--epochs` / `--batch-size` / `--lr` / `--early-stopping-patience` 等与
  六折 CV 运行一致；DeepDTAGen 默认 batch=32、eval batch=128。

## 输出

每个运行写到一个独立目录
`<results-root>/<Model>_<dataset>_<setting>_fold<N>_<时间戳>/`，包含
`config.json`、`split_indices.json`、`history.csv`、Top-3 检查点、
`validation_summary.json`、`test_metrics.json`（best_checkpoint 与
top3_average 两套指标）和 `test_predictions.npz`。

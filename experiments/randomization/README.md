# 随机化对照实验（六折划分）

## 协议

随机化实验沿用样本级六折轮转协议：
第 N 折测试集 = 第 N 折样本，验证集 = 第 (N+1) mod 6 折样本，训练集 = 其余
4 折样本（与 `splits/<dataset>/fold_<N>.json` 完全一致）。在训练集上施加
可复现的随机置换：

- `rand_x1`：置换训练样本的**蛋白（靶点）特征**，破坏"化合物-蛋白"对应；
- `rand_x2`：置换训练样本的**化合物图结构**（x / edge_index / c_size），
  破坏输入特征与标签的对应；
- `rand_y`：置换训练样本的**亲和力标签**，破坏标签与输入特征的对应。

验证集与测试集始终保持不变，因此指标衡量的是模型在被打乱的训练集上还能
学到多少真实结合规律（对照实验）。

## 1. 生成数据集

```bash
cd experiments/randomization
python prepare_randomization.py --datasets davis kiba bindingdb
```

输出：

- `data/splits/<dataset>/rand_x1/fold_0..5.json`
- `data/splits/<dataset>/rand_x2/fold_0..5.json`
- `data/splits/<dataset>/rand_y/fold_0..5.json`

每个清单包含 `train_permutation`（按位置与 `train_indices` 对齐的置换表）、
三个集合的样本数以及置换有效性审计。

## 2. 训练

```bash
cd experiments/randomization

# 单个折（示例）
python train_random.py \
  --dataset davis --mode x2 --fold 0 --model combined_dta_edge

# davis + kiba + bindingdb、x1/x2/y、全部 6 折
bash run_randomization.sh --datasets davis kiba bindingdb
```

## 常用参数

- `--dry`：2 个 epoch 冒烟测试。
- `--results-root <路径>`：默认 `experiments/results/randomization`。
- `--skip-done`：断点续跑时跳过已完成运行。
- `--model`：默认 `combined_dta`，可换任意 `models/` 下的 MDS 模型。

## 输出

与冷启动相同的结果布局（config / split_indices / history / top3 检查点 /
validation_summary / test_metrics / test_predictions.npz）。

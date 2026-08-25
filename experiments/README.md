# 六折划分下的三类补充实验

| 文件夹 | 实验 | 划分方式 | 设置 |
| --- | --- | --- | --- |
| `cold_start/` | 冷启动 | 实体级六折 | `cold_drug`、`cold_target`（两个单冷）+ `cold_both`（双冷） |
| `randomization/` | 随机化对照 | 样本级六折 | `x1`（靶点特征）、`x2`（化合物图）、`y`（标签） |
| `fewshot/` | 少样本 | 样本级六折 | `fs50`（50%）、`fs25`（25%）、`fs10`（10%） |

每个文件夹内：

- `prepare_*.py`：从 `<dataset>_sixfold_all.csv` 和
  `splits/<dataset>/fold_membership.json` 生成对应实验的六折数据集
  （清单 JSON 与审计），不复制多 GB 的 .pt 文件；
- `train_*.py`：训练入口（验证集早停 + Top-3 平均 + 一次性测试评估），
  结果布局与六折 CV 完全一致；
- `run_*.sh`：一键顺序运行全部设置/折数（支持 `--dry` 冒烟与
  `--skip-done` 断点续跑）；
- `README.md`：具体运行指令。

只有 `cold_start/` 需要兼容 DeepDTA / GraphDTA / DeepDTAGen：

```bash
cd experiments/cold_start
bash run_cold_start.sh --model deepdta --datasets davis kiba bindingdb
bash run_cold_start.sh --model graphdta_gcn --datasets davis kiba bindingdb
bash run_cold_start.sh --model deepdtagen --datasets davis kiba bindingdb
```

随机化与少样本按论文口径运行 MDS 模型族（`--model combined_dta` 系列）。

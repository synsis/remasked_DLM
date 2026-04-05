# Evaluation Status (results_v2)

## 已从 tex 删除的数据集（完全没跑 or 没跑完）

| Benchmark | 论文分类 | 状态 | 删除原因 |
|-----------|---------|------|---------|
| **OlympiadBench** | Math | 完全没跑 | results_v2 无目录 |
| **MBPP+** | Coding | 完全没跑 | results_v2 无目录 |
| **MMLU-Pro** | Knowledge | 未完成 | original ~3270/12032, remask ~1410/12032 |
| **BBH** | Reasoning | Remask 未完成 | original done, remask ~4960/6511 |
| **GSM-Plus** | Math | Remask 未跑 | original done (69.5%), remask 未启动 |

## 已从 tex 删除的 Strategy 行（主结果表）

主结果表只保留 Original (T2T) vs T2M Remask 两行。
三种 strategy (LowProb, T2T-Remask, LogitDiff) 保留在 ablation 表中。

| Strategy | 状态 |
|----------|------|
| **T2T-Remask** | 全线未跑（ablation 表保留占位） |
| **LogitDiff** | 全线未跑（ablation 表保留占位） |

## 论文当前保留的 8 个 benchmark（均已跑完 original + remask）

| Benchmark | 分类 | Original | T2M Remask | Delta |
|-----------|------|----------|------------|-------|
| GPQA | Knowledge | 21.7 (acc) | 22.7 | +1.0 |
| TriviaQA | Knowledge | 14.2 (EM) / 30.7 (F1) | 14.9 / 31.9 | +0.8 / +1.1 |
| HellaSwag | Reasoning | 81.3 (acc) | 81.2 | -0.2 |
| PIQA | Reasoning | 79.2 (acc) | 79.1 | -0.2 |
| DROP | Reasoning | 45.0 (EM) / 54.7 (F1) | 46.0 / 55.3 | +1.0 / +0.6 |
| CMATH | Math | 82.9 (acc) | 88.7 | +5.8 |
| AIME 2025 | Math | 33.3 (acc) | 40.0 | +6.7 |
| HumanEval+ | Coding | 64.6 (pass@1+) | 64.6 | 0.0 |

## 跑了但论文没用

| Benchmark | 备注 |
|-----------|------|
| **IFEval** | 两边跑完，prompt_strict 73.0→73.4，可考虑加入 |

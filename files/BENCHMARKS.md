# LLaDA2.1 Remask — Benchmark & Hyperparameter Reference

## 1. 推理超参数（来自官方 README + 代码）

### 1.1 官方推荐参数

| 参数 | Q Mode | S Mode | 代码默认 | 说明 |
|------|--------|--------|----------|------|
| `threshold` | **0.7** | 0.5 | 0.95 | M2T: mask 位 softmax 概率 > 此值才填 token |
| `editing_threshold` | **0.5** | 0.0 | 0.9 | T2T: 新 token 概率 > 此值且与旧不同才替换 |
| `max_post_steps` | 16 | 16 | 16 | 所有 mask 填完后的额外迭代次数（用于 T2T edit） |
| `block_length` | 32 | 32 | 32 | 每个 block 的并行解码窗口大小 |
| `temperature` | 0.0 | 0.0 | 0.0 | 采样温度，0=greedy |
| `gen_length` | 16384 | 16384 | 2048 | 最大生成 token 数（配合 eos_early_stop） |
| `eos_early_stop` | True | True | False | 遇到 EOS 提前停止 |
| `top_p` | None | None | None | 核采样 |
| `top_k` | None | None | None | top-k 采样 |

> **注意**: `steps` 参数（代码默认 32）在当前 LLaDA2.1 代码中 **实际未被使用**。
> 内循环是 `while True` 直到收敛（所有 mask 填完 + 无 edit），
> 不是 `for step in range(steps)`。设 32 或 1 效果一样。

### 1.2 block_length=32 为什么是对的

block_length 是并行 denoise 窗口大小，不是总序列长度。生成 N 个 token 需要 N/32 个 block，
每个 block 内 32 个位置迭代 denoise。小 block 保证位置间独立性，是 block diffusion 的设计。

三方验证：
- 官方 README: "We recommend `block_length=32`"
- HuggingFace Diffusers: "block_length=32 ... are the universal inference parameters"
- SGLang Cookbook: 所有部署示例均用 block_length=32

### 1.3 gen_length 选择

官方推荐 16384，但单卡跑太慢。实际选择原则：
- `eos_early_stop=True` 时 gen_length 只是上界，EOS 出现即停
- 设太大会浪费 attention mask 显存（16384² × 2B ≈ 512MB）
- 按任务类型设合理上界即可

### 1.4 我们的 Remask 特有参数

| 参数 | 策略1: low_prob | 策略2: t2t_remask | 策略3: logit_diff |
|------|----------------|-------------------|-------------------|
| `remask_threshold` 默认 | **0.5** | 0.9 | 0.3 |
| 含义 | P(old_token\|ctx) < 0.5 → remask | P(new_token) > 0.9 且 new≠old → remask | \|ΔP(old)\| > 0.3 → remask |
| `max_remask_per_pos` | 5 | 5 | 5 |

---

## 2. 全部 Benchmark（论文 33 个）

### Knowledge (5)

| Benchmark | HF Dataset | Split | 样本数 | Metric | gen_length | 说明 |
|-----------|-----------|-------|--------|--------|------------|------|
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | test | 12,032 | Accuracy | 1024 | 10选项多选题 |
| GPQA-Diamond | `Idavidrein/gpqa` (gpqa_diamond) | train | 198 | Accuracy | 1024 | ⚠️ Gated，需 HF token |
| C-Eval | `ceval/ceval-exam` (per subject) | val | ~1,346 | Accuracy | 512 | 中文学科考试，需逐科目加载 |
| PHYBench | `Phy-Bench/PHYBench` | test | ~? | Math match | 1024 | 可能需本地数据 |
| TriviaQA | `trivia_qa` (rc.nocontext) | validation | 17,944 | EM / F1 | 256 | 短答案 QA |

### Reasoning (12)

| Benchmark | HF Dataset | Split | 样本数 | Metric | gen_length | 说明 |
|-----------|-----------|-------|--------|--------|------------|------|
| BIG-Bench Hard | `lukaemon/bbh` (per task) | test | ~6,500 | EM | 512 | 27个子任务，需逐个加载拼接 |
| BIG-Bench Extra Hard | `BBEH/bbeh` | train | 4,520 | EM | 512 | |
| bbh-zh | `opencompass/bbh_zh` | test | ~6,500 | EM | 512 | BBH 中文版 |
| MuSR | `TAUR-Lab/MuSR` | 3 splits | 756 | Accuracy | 512 | murder/object/team 三个 split |
| ZebraLogic | 需查找 | test | ~? | Accuracy | 1024 | 逻辑推理 |
| PrOntoQA | `renma/PrOntoQA` | validation | 500 | Accuracy | 256 | True/False 判断 |
| PIQA | `lmms-lab/piqa` | test | 1,838 | Accuracy | 256 | 2选1 |
| OCNLI | `clue` (ocnli) | validation | 2,950 | Accuracy | 256 | 中文 NLI 三分类 |
| HellaSwag | `Rowan/hellaswag` | validation | 10,042 | Accuracy | 256 | 4选1 续写 |
| KOR-Bench | 需查找 | test | ~? | Accuracy | 512 | 可能需本地数据 |
| DROP | `ucinlp/drop` | validation | 9,535 | EM / F1 | 256 | 阅读理解 |
| SQuAD 2.0 | `rajpurkar/squad_v2` | validation | 11,873 | EM / F1 | 256 | 含无答案问题 |

### Coding (8)

| Benchmark | HF Dataset | Split | 样本数 | Metric | gen_length | 说明 |
|-----------|-----------|-------|--------|--------|------------|------|
| HumanEval+ | 本地 `data/HumanEvalPlus.jsonl` | - | 164 | pass@1 | 512 | 需 evalplus 评分 |
| MBPP+ | 本地 `data/MbppPlus.jsonl` | - | 378 | pass@1 | 512 | 需 evalplus 评分 |
| CRUXEval-O | 需查找 | test | ~800 | EM | 256 | 预测代码输出 |
| MultiPL-E | - | - | - | pass@1 | 512 | ⚠️ 需多语言编译器 |
| BigCodeBench-Full | - | - | ~1,140 | pass@1 | 1024 | ⚠️ 需代码执行沙箱 |
| LiveCodeBench | - | - | ~? | pass@1 | 1024 | ⚠️ 需代码执行 |
| Spider | - | - | ~1,034 | Execution Acc | 512 | ⚠️ 需数据库执行 |
| BIRD-SQL | - | - | ~1,534 | Execution Acc | 512 | ⚠️ 需数据库执行 |

### Math (5)

| Benchmark | HF Dataset | Split | 样本数 | Metric | gen_length | 说明 |
|-----------|-----------|-------|--------|--------|------------|------|
| GSM-Plus | `qintongli/GSM-Plus` | testmini | 2,400 | EM (numeric) | 512 | GSM8K 加强版 |
| CMATH | `weitianwen/cmath` | test | 1,098 | EM (numeric) | 512 | 中文小学数学 |
| AIME 2025 | 本地 `data/aime2025.json` | - | 30 | EM (integer) | 1024 | 竞赛数学，答案 0-999 |
| OlympiadBench | `math-ai/olympiadbench` | test | 674 | Math match | 1024 | 奥赛级数学 |
| Omni-MATH | `KbsdJames/Omni-MATH` | test | 4,428 | Math match | 1024 | |

### Agent & Alignment (3)

| Benchmark | HF Dataset | Split | 样本数 | Metric | gen_length | 说明 |
|-----------|-----------|-------|--------|--------|------------|------|
| IFEval | `google/IFEval` | train | 541 | strict-prompt acc | 1024 | 需离线 ifeval 评分 |
| BFCL v3 | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | test | ~? | 需离线评分 | 512 | 函数调用 |
| Nexus FC | 需查找 | - | ~? | 需离线评分 | 512 | 函数调用 |

---

## 3. 数据集可用性总结

| 状态 | 数量 | 说明 |
|------|------|------|
| ✅ 可直接运行 | 19 | HF 公开数据 + 本地数据 |
| 🔑 需 HF Token | 1 | GPQA (gated dataset) |
| 🔧 需外部工具 | 8 | MultiPL-E, BigCodeBench, LiveCodeBench, Spider, BIRD, BFCL, Nexus FC, 部分需编译器/数据库 |
| ❓ 需查找数据源 | 5 | ZebraLogic, KOR-Bench, CRUXEval, PHYBench, bbh-zh（部分可能需本地数据） |

## 4. 调参计划

### 4.1 小样本快速调参（每个 benchmark 取 20-50 条）

优先选择小且快且多领域的 benchmark：

| Benchmark | 领域 | 样本数 | 每条约耗时 | 20条总耗时 |
|-----------|------|--------|-----------|-----------|
| PIQA | reasoning | 1,838 | ~10s | ~3min |
| HellaSwag | reasoning | 10,042 | ~10s | ~3min |
| CMATH | math (CN) | 1,098 | ~30s | ~10min |
| GSM-Plus | math | 2,400 | ~40s | ~13min |
| MMLU-Pro | knowledge | 12,032 | ~30s | ~10min |

### 4.2 调参网格

| 策略 | 阈值候选 |
|------|---------|
| low_prob | 0.3, **0.5**, 0.7 |
| t2t_remask | 0.7, 0.8, **0.9** |
| logit_diff | 0.1, 0.2, **0.3** |

### 4.3 对比实验

每组实验跑 `original` (Q Mode) 和 `remask` 对比，保持其他超参完全一致：
- threshold=0.7, editing_threshold=0.5, block_length=32, temperature=0.0

---

## 5. LLaDA2.1-mini 论文报告指标（Q Mode，供参考对比）

| Benchmark | Q Mode Score |
|-----------|-------------|
| GSM-Plus | 86.55 |
| MMLU-Pro | 64.84 |
| HumanEval+ | 82.93 |
| MBPP+ | 74.07 |
| CMATH | 94.99 |
| HellaSwag | 76.19 |
| PIQA | 86.89 |
| DROP | 82.37 |
| AIME 2025 | 43.33 |
| IFEval | 83.18 |

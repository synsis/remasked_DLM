# T2M Remasking 论文改进计划（NeurIPS 2026）

> 基于当前论文状态和已有实验数据的系统性分析
> 更新日期：2026-03-29

---

## 当前实验数据快照

| Benchmark | Original (T2T) | LowProb (T2M) | Delta | Forward Pass 倍数 | 状态 |
|-----------|----------------|----------------|-------|-------------------|------|
| CMATH | 82.88% | **88.71%** | **+5.83%** | 2.3x | ✅ 完成 |
| PIQA | 79.22% | 79.05% | -0.16% | 3.9x | ✅ 完成 |
| DROP | F1=27.85% | — | — | — | ⚠️ original 未跑完 |
| HellaSwag | 71.56% | — | — | — | ⚠️ original 未跑完 (960/10042) |
| TriviaQA | EM=14.54% | — | — | — | ⚠️ original 未跑完 |
| AIME 2025 | 8 samples | 0 samples | — | — | ⚠️ 几乎未开始 |
| MMLU-Pro | — | — | — | — | ❌ 未开始 |
| GPQA | — | — | — | — | ❌ 未开始 |
| BBH | — | — | — | — | ❌ 未开始 |
| GSM-Plus | — | — | — | — | ❌ 未开始 |
| HumanEval+ | — | — | — | — | ❌ 未开始 |
| MBPP+ | — | — | — | — | ❌ 未开始 |

**只有 LowProb 策略跑了 2 个 benchmark，T2T-Remask 和 LogitDiff 策略 0 个结果。**

### 关键信号

- **CMATH +5.83%**：数学推理场景 T2M 确实有效，这是核心卖点
- **PIQA -0.16%**：分类/短生成场景无收益，且 overhead 显著
- **Inference overhead**：forward pass 增加 2.3x–3.9x，wall-clock 增加约 1.9x–3.5x

---

## Tier 1: 不做就别投（Must-have）

### 1. 补全实验矩阵

**问题**：表格全是 `--`，reviewer 直接 reject。

**最低要求**：
- 3 策略 × 至少 6–8 个 benchmark 的完整结果
- DROP / HellaSwag / TriviaQA 的 original baseline 需要跑完（当前 `done: false`）
- AIME 2025 需要完整跑完（目前只有 8 个 original sample）

**优先级排序**（按预期 impact 从高到低）：
1. 数学类：GSM-Plus, CMATH(done), AIME 2025 — 预期收益最大
2. 推理类：BBH, DROP — 长链推理，期望有收益
3. 知识类：MMLU-Pro, GPQA, TriviaQA — 检验泛化性
4. 编程类：HumanEval+, MBPP+ — 代码生成，结构化输出
5. 分类类：HellaSwag, PIQA — 预期收益小，但要有数据

### 2. 加 Figure 1：方法概览图

**问题**：当前论文 **0 张图**。NeurIPS 论文没有图就失去了视觉锚点，reviewer 第一印象差。

**方案**：一张 T2T vs T2M 的对比流程图

```
Figure 1 设计:

上半（T2T）:
  [the] [cat] [sat] [purple] [mat]     ← 初始状态，"purple" 是错误 token
       ↓ T2T 检测到 "purple" 不对
  [the] [cat] [sat] [on▲] [mat]        ← 替换为 "on"，但可能也不正确
       ↓ 下一迭代
  后续位置被 "on" 的错误语义误导...     ← 上下文污染传播

下半（T2M）:
  [the] [cat] [sat] [purple] [mat]     ← 相同初始状态
       ↓ T2M 检测到 "purple" 不对
  [the] [cat] [sat] [M] [mat]          ← 重置为 mask（中性信号）
       ↓ M2T 在干净上下文下重预测
  [the] [cat] [sat] [on✓] [mat]        ← 联合重优化得到正确结果
```

右侧加上对应的理论概念标注：correction inertia / context purification / delayed commitment。

### 3. 直面 inference overhead

**问题**：reviewer 一定会质疑 "你用 2x compute 换 6% accuracy，值不值？"

**应对方案**：
- 在 Analysis 节报告 overhead 数据并诚实讨论
- 加 **accuracy-vs-compute scatter plot**（x = forward passes, y = accuracy），论证 T2M 在 accuracy-compute tradeoff 上更优
- 探索优化手段：
  - 对 `C_max` 更激进地收缩（如 C_max=2）
  - 前几次迭代不做 remasking（让 M2T 先稳定）
  - 迭代后期降低 remasking rate

---

## Tier 2: 做了大幅提升竞争力

### 4. Token 去噪轨迹可视化

**为什么重要**：这会是论文**最出彩**的 figure，直接视觉化证明核心论点。

**设计**：对单个数学题样本，画热力图：
- X 轴 = 去噪迭代步
- Y 轴 = block 内 token 位置
- 颜色编码 = mask（灰）/ 正确 token（绿）/ 错误 token（红）/ 被 remask（蓝闪）

**对比**：T2T 下某位置 stuck 在红色（correction inertia），T2M 下该位置 红→蓝→绿（remask → re-predict → correct）。

**实现**：需要在 `_token_edit_step` 中加 logging，记录每步每位置的 token ID 和 ground truth 对比。

### 5. 加 No-Edit baseline

**为什么重要**：隔离 T2T 本身的效果。

**三个 baseline**：
| 方法 | 说明 |
|------|------|
| No-Edit | 完全关闭 T2T/T2M，只用 M2T |
| Original (T2T) | LLaDA2.1 原始设置 |
| T2M (3 策略) | 我们的方法 |

**可能的发现**：
- 如果 No-Edit ≈ T2T：说明 T2T 本身效果有限，我们的工作更有意义
- 如果 No-Edit < T2T < T2M：说明编辑有用，T2M 是更好的编辑方式
- 如果 No-Edit > T2T（某些 benchmark）：说明 T2T 可能有害，T2M 的"移除错误信号"论点更强

**实现**：改一行代码 — `editing_threshold = 1.0`（永远不触发 T2T）。

### 6. 策略组合实验

**动机**：三个策略检测不同类型的错误，组合可能互补。

**组合方式**：
- **Union**: LowProb ∪ LogitDiff — 任一触发就 remask（高召回）
- **Intersection**: 两者都触发才 remask（高精度）
- **Cascade**: LowProb 粗筛 + LogitDiff 确认

如果组合 > 单策略最优，说明策略确实互补，是很好的 contribution。

### 7. 任务类型分析

**核心 insight**：解释为什么 CMATH 涨但 PIQA 不涨。

**假说**：T2M 收益与以下因素正相关：
- 生成长度（更长 → 更多错误传播机会）
- 推理链复杂度（数学/代码 > 知识问答 > 选择题）
- 原始错误率（错误越多 → remask 纠正空间越大）

**分析方法**：
- 按 avg_output_tokens 分层统计 delta
- 画 scatter：x = 生成长度, y = T2M 相对提升

如果能证明 **"任务越难/生成越长，T2M 收益越大"**，这是非常有力的 insight。

### 8. Compute-fair comparison

**设计**：给 original T2T 也多跑 denoising steps（如 steps=64 而非 32），或放宽 T2T 阈值。在**相同 forward pass 预算**下比较 accuracy。

如果 T2M 在 equal-compute 下依然更好，彻底堵住 reviewer 的 overhead 质疑。

### 9. 阈值敏感性改成图

当前 ablation 是表格形式。改为折线图：
- x 轴 = threshold 值
- y 轴 = accuracy
- 每条线一个策略

图比表格直观 10x，占空间更少，reviewer 一眼看到最优区间。

---

## Tier 3: 锦上添花

### 10. Per-block 分析

统计 remask 在第几个 block 发生最多：
- 前面 block remask 多 → 初始不确定性高
- 后面 block remask 多 → exposure bias 累积

给出 mechanistic insight，增加分析深度。

### 11. Oracle upper bound

做 cheating experiment：只 remask 那些确实错误的 token（对比 ground truth）。这给出 T2M 的理论上限：
- Oracle 远高于当前策略 → 更好的检测策略是明确的未来方向
- Oracle ≈ 当前策略 → 当前策略已经够好

### 12. 扩展到其他 dLLM

如果 Dream 有类似编辑机制，在 Dream 上跑 1–2 个 benchmark，论文 scope 从 "LLaDA2.1 specific" 提升到 "general dLLM principle"。

---

## 执行计划

### Phase 1: 补数据（最紧急，1–2 周）

```
并行跑：
├── Original baseline 补完：DROP, HellaSwag, TriviaQA, AIME, GSM-Plus, MMLU-Pro, GPQA, BBH, HumanEval+, MBPP+
├── LowProb remask：除 CMATH/PIQA 外所有 benchmark
├── T2T-Remask：全部 benchmark
├── LogitDiff：全部 benchmark
└── No-Edit baseline：全部 benchmark
```

### Phase 2: 核心图表（与 Phase 1 并行，1 周）

```
├── Figure 1: T2T vs T2M 方法概览图（TikZ）
├── Figure 2: Token 去噪轨迹热力图（需加 logging 代码）
├── Figure 3: Accuracy vs Forward Passes scatter plot
└── Figure 4: 阈值敏感性折线图
```

### Phase 3: 深度分析（Phase 1 完成后，0.5–1 周）

```
├── 策略组合实验
├── 任务类型分析（生成长度 vs delta 相关性）
├── Per-block remask 分布分析
├── Compute-fair comparison
└── 更新论文中英文版本，填入所有数据和图表
```

### Phase 4: 收尾（0.5 周）

```
├── 完善 Related Work 定位
├── 补 Appendix 案例研究
├── 填写 NeurIPS Checklist
├── 最终一致性检查（中英文同步）
└── 准备代码和 supplementary material
```

---

## 风险评估

| 风险 | 影响 | 应对 |
|------|------|------|
| 多数 benchmark T2M 无提升 | 致命 | 如果只有数学类有效，重新定位为 "math reasoning specific"；或探索更好的阈值/策略 |
| Overhead 太大 reviewer 不接受 | 高 | 加 compute-fair comparison，优化 remasking efficiency |
| T2T-Remask 和 LogitDiff 效果差 | 中 | 至少保证 LowProb 全面有效；如果只有一个策略有效，减少策略数聚焦深度 |
| No-Edit 比 T2T 好 | 低风险高收益 | 如果成立，这反而是更强的论点——T2T 有害 |
| AIME 等困难 benchmark 效果不显著 | 中 | 这些 baseline 本身分数就低，delta 可能被噪声淹没；考虑 majority voting |

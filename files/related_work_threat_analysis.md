# Related Work & 威胁分析：LLaDA2.1 T2M Remask Inference

> 调研时间：2026-03-29

## 1. 我们的工作概述

在 LLaDA2.1 的推理过程中，将 **T2T (Token-to-Token) editing** 替换为 **T2M (Token-to-Mask) remasking**。

核心思想：在每个 block 的迭代去噪中，editing 阶段不再将可疑 token 直接替换为另一个 token（T2T），而是将其退回 `<mask>`（T2M），让后续迭代在更干净的 context 下重新预测。

三种 remask 策略：
- **low_prob**：当前 logits 对旧 token 预测概率低 → remask
- **t2t_remask**：模型高置信地想换一个不同 token → 旧位置 remask
- **logit_diff**：新旧两步 logits 概率差异大 → remask

Training-free，开箱即用，无需额外训练。

---

## 2. 高威胁论文

### 2.1 ReMDM — Remasking Discrete Diffusion Models with Inference-Time Scaling

| 属性 | 内容 |
|------|------|
| arXiv | [2503.00307](https://arxiv.org/abs/2503.00307) (2025.03) |
| 发表 | **NeurIPS 2025**（已接收） |
| 作者 | Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, Volodymyr Kuleshov (Cornell) |
| 代码 | [kuleshov-group/remdm](https://github.com/kuleshov-group/remdm), [ReMDM-LLaDA](https://github.com/guanghanwang/ReMDM-LLaDA) |

**核心做法**：修改 masked diffusion 的 **reverse posterior**，引入 remasking 概率 $\sigma_t$：

$$q_\sigma(z_s | z_t, x) = \begin{cases} \text{Cat}(z_s;\; (1-\sigma_t) \cdot x + \sigma_t \cdot \text{mask}), & z_t \neq \text{mask} \\ \text{Cat}(z_s;\; \text{原分布}), & z_t = \text{mask} \end{cases}$$

当 $z_t$ 已经是 unmasked token 时，有 $\sigma_t$ 概率被**随机**退回 mask。证明此修改不改变前向 marginal $q(z_t|x)$，可直接复用预训练权重。

**多种 schedule 设计**：
- **ReMDM-cap**：$\sigma_t = \min(\eta_{cap}, \sigma_t^{max})$，给 remask 概率加上限
- **ReMDM-rescale**：$\sigma_t = \eta_{rescale} \cdot \sigma_t^{max}$，按比例缩放
- **ReMDM-conf**：置信度低的 token 被 remask 概率更高（confidence-based）
- **ReMDM-loop**：三阶段——正常 decode → 固定 $\alpha_t$ 做反复 remask loop → 收尾

**与我们的关键区别**：

| 维度 | 我们的 T2M Remask | ReMDM |
|------|-------------------|-------|
| 作用位置 | 只在 LLaDA2.1 的 **T2T editing step** | 改变**整个 reverse sampling process** 的每一步 |
| 理论基础 | Heuristic（token probability / logit diff） | 概率框架（修改 reverse posterior） |
| 决策方式 | **Targeted**：显式判断哪些 token 有问题 | **Random**：每个 token 以概率 $\sigma_t$ 被随机 remask |
| 适用范围 | 专门针对 LLaDA2.1 block-level 生成 | 通用 MDLM 框架 |

**威胁评估**：🔴 **最高**。核心 idea "把不好的 token 退回 mask" 高度重叠，已发表顶会，已有 LLaDA 实现。

**我们的差异化**：
1. ReMDM 是 random remask，我们是 targeted remask——不浪费算力重新预测本来正确的 token
2. ReMDM 没有利用 LLaDA2.1 的 T2T editing 结构
3. 我们的理论分析（训练-推理分布对齐、错误传播、条件熵）是 ReMDM 未讨论的角度

---

### 2.2 CORE — Context-Robust Remasking for Diffusion Language Models

| 属性 | 内容 |
|------|------|
| arXiv | [2602.04096](https://arxiv.org/abs/2602.04096) (2026.02) |
| 作者 | Kevin Zhai, Sabbir Mollah, Zhenyi Wang, Mubarak Shah |

**核心做法**：Training-free 的 remasking 框架。不信任静态 token 概率，而是通过**上下文扰动**检测 "context-brittle" token——即对 conditioning context 的 masked perturbation 敏感的 token。将 revision 形式化为鲁棒优化问题。

**在 LLaDA-8B 上测试**：MBPP 提升 +9.2%。

**威胁评估**：🔴 **高**。**直接批评了 "low probability heuristics"**，认为基于静态置信度/低概率的 remask 策略是 "inherently myopic"（本质短视）。这正是我们 `low_prob` 策略的做法。

**应对**：
1. 我们的 `logit_diff` 策略（看两步之间概率变化）某种程度上也捕捉了上下文动态变化，可以与 CORE 做对比讨论
2. CORE 需要额外的 perturbation forward pass，计算开销更大；我们的方法是 zero-overhead 的

---

### 2.3 RemeDi — Don't Settle Too Early: Self-Reflective Remasking

| 属性 | 内容 |
|------|------|
| arXiv | [2509.23653](https://arxiv.org/abs/2509.23653) (2025.09) |
| 已有 9B 模型 | [maple-research-lab/RemeDi-RL](https://huggingface.co/maple-research-lab/RemeDi-RL) |

**核心做法**：Dual-stream 架构——Token Prediction Stream (TPS) + Unmasking Policy Stream (UPS)。UPS 输出 per-token confidence score 决定哪些 token 需要 remask。两阶段训练：Remask SFT + Remask RL。

**威胁评估**：🔴 **高**。和我们一样做 remasking，但通过训练让模型学习什么时候该 remask，比 heuristic 更 principled。

**我们的优势**：Training-free，无需额外训练，开箱即用。

---

### 2.4 ProSeCo — Learn from Your Mistakes: Self-Correcting Masked Diffusion Models

| 属性 | 内容 |
|------|------|
| arXiv | [2602.11590](https://arxiv.org/abs/2602.11590) (2026.02) |
| 作者 | Cornell（与 ReMDM 同组） |

**核心做法**：训练 MDM 同时学习 decode 和 correct。在标准 MDM loss 上加 cross-entropy loss，让模型从自己的错误中学习纠错。Inference 时交替执行 unmasking 和 corrective refinement。

**结果**：2-3x 更快 sampling，LLaDA 8B 上 HumanEval, MBPP, GSM8K, Minerva 大幅提升。

**威胁评估**：🔴 **高**。ReMDM 的升级版，在我们同样的 benchmark 上有结果。

---

## 3. 中等威胁论文

### 3.1 PRISM — Fine-Tuning Masked Diffusion for Provable Self-Correction

| 属性 | 内容 |
|------|------|
| arXiv | [2510.01384](https://arxiv.org/abs/2510.01384) (2025.10) |
| 发表 | ICLR 2026 Workshop |

轻量 adapter fine-tuning，学习 per-token quality score（近似 $\Pr[x_i = y_i \mid y \oplus m_i]$）。在 LLaDA 8B MBPP 上有结果。需要 fine-tuning。

### 3.2 RCR (Running Confidence Remasking) in MDPO

| 属性 | 内容 |
|------|------|
| arXiv | [2508.13148](https://arxiv.org/abs/2508.13148) (2025.08) |

Training-free 的 remasking 策略，用 running confidence scores 做 remask。MDPO + RCR 在 MATH500 提升 9.6%，Countdown 提升 54.2%。

### 3.3 ReMix — Rejection Mixing

| 属性 | 内容 |
|------|------|
| arXiv | [2602.22868](https://arxiv.org/abs/2602.22868) (2026.02) |

用连续表示做 iterative refinement，rejection rules 把不确定 token 退回 mask。Training-free，声称 2-8x 推理加速。

### 3.4 IterRef — Effective Test-Time Scaling via Iterative Refinement

| 属性 | 内容 |
|------|------|
| arXiv | [2511.05562](https://arxiv.org/abs/2511.05562) (2025.11) |

Reward-guided noising-denoising refinement，Multiple-Try Metropolis 框架，有收敛保证。

---

## 4. 较低威胁（但需关注）

| 论文 | arXiv | 方向 |
|------|-------|------|
| LLaDA-o | [2603.01068](https://arxiv.org/abs/2603.01068) (2026.03) | LLaDA 多模态扩展，方向不同 |
| Token Ordering | [2502.06768](https://arxiv.org/abs/2502.06768) (2025.02) | 改进 unmask 顺序而非 remask |

---

## 5. 竞争格局总结

```
时间线：
2025.03  ReMDM (NeurIPS 2025)         ← 首发 remasking idea
2025.08  MDPO + RCR                    ← training-free remasking
2025.09  RemeDi                        ← 训练 confidence score 做 remasking
2025.10  PRISM                         ← lightweight adapter 学 quality score
2025.11  IterRef                       ← reward-guided refinement
2026.02  ProSeCo                       ← ReMDM 同组升级版，加训练
2026.02  CORE                          ← 批评 low-prob heuristic，用 context perturbation
2026.02  ReMix                         ← 连续空间 rejection-based
2026.03  我们                          ← targeted T2M remask 替换 LLaDA2.1 T2T editing
```

**结论**："Remasking" 已是一个非常拥挤的赛道，从 2025.03 到现在有 7-8 篇直接相关论文。

---

## 6. 我们的差异化定位

### 6.1 已有的差异点

1. **Targeted vs Random**：我们显式检测错误 token 做定向 remask，不像 ReMDM 的随机 remask
2. **聚焦 T2T editing step**：不改变整体 sampling，只替换 LLaDA2.1 特有的 T2T editing 机制
3. **理论分析角度独特**：训练-推理分布对齐、错误 context 传播、条件熵分析（T2T vs T2M）是其他论文未深入讨论的
4. **Training-free**：相比 RemeDi、ProSeCo、PRISM 无需额外训练
5. **多策略对比**：同时实验 low_prob / t2t_remask / logit_diff 三种策略

### 6.2 最大风险

1. CORE 直接批评了 low-prob heuristics —— 命中我们主策略
2. ReMDM 已发 NeurIPS 2025，核心 idea 有 prior art
3. 如果实验结果没有在众多 benchmark 上一致超过 ReMDM/CORE，纯方法创新难以说服 reviewer

### 6.3 建议行动

1. **务必做 ReMDM 对比实验**：他们已有 [ReMDM-LLaDA](https://github.com/guanghanwang/ReMDM-LLaDA) 代码，在 LLaDA2.1 上跑对比
2. **做深 logit_diff 策略**：它比 low_prob 更难被 CORE 的批评覆盖（看的是动态变化而非静态概率）
3. **强调 T2T editing 的特殊性**：LLaDA2.1 官方引入的 T2T editing 本身就是一个 post-hoc refinement 机制，我们是在这个机制内做改进，不是 general-purpose remasking
4. **尽快跑完全部 33 个 benchmark**：拿到和竞品可比的全面数据
5. **考虑与 ReMDM 结合**：在 sampling 层面用 ReMDM，在 editing 层面用我们的 T2M，看是否互补

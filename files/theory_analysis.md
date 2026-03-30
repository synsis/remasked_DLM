# 理论分析：为什么 Remask (T2M) 优于 Token Edit (T2T)

## 1. 背景与符号

LLaDA 是一个离散扩散（masked diffusion）语言模型，生成过程是 block-level 的迭代去噪：每个 block 内，模型通过多步迭代将 `<mask>` token 逐步替换为真实 token。每步包括两个阶段：

- **M2T（Mask → Token）**：对仍为 `<mask>` 的位置，选出模型置信度高的 token 写入
- **编辑阶段（对已 unmask 的 token）**：检查已生成的 token 是否需要修正

编辑阶段是本文分析的核心。设：
- \( x = (x_1, \ldots, x_n) \) 为当前 block 内的 token 序列
- \( x_i^{\text{old}} \) 为位置 \( i \) 当前已 commit 的 token
- \( p_\theta(x_i \mid x_{-i}) \) 为模型给出的条件分布
- \( x_i^{\ast} \) 为采样的 argmax token：\( x_i^{\ast} = \arg\max p_\theta(x_i \mid x_{-i}) \)

**T2T (Token Edit)**：当 \( p_\theta(x_i^{\ast} \mid x_{-i}) > \tau \) 且 \( x_i^{\ast} \neq x_i^{\text{old}} \) 时，直接将位置 \( i \) 替换为 \( x_i^{\ast} \)。

**T2M (Remask)**：当某个信号表明 \( x_i^{\text{old}} \) 不正确时，将位置 \( i \) 重置为 `<mask>`，让后续迭代重新预测。

---

## 2. T2T 的置信度-改变耦合问题

### 2.1 核心问题

T2T 的触发条件将两个本质上独立的判断**耦合**在一起：

1. **错误检测**：当前 token 是否错误？
2. **替代确定**：是否存在一个确切的高置信度替代？

T2T 要求两个条件**同时**满足才行动。但在实践中，这两个事件的发生并不总是同步的。

### 2.2 Case A：旧 token 明确错误，但无单一高置信候选

考虑一个语义位置，其真实分布是多模态的。例如"他感到非常____"，"高兴"、"开心"、"愉快"都是合法答案。假设当前 token 是完全不相关的"紫色"。

此时模型的预测分布可能为：

\[
p_\theta(\text{高兴}) = 0.28, \quad p_\theta(\text{开心}) = 0.25, \quad p_\theta(\text{愉快}) = 0.20, \quad p_\theta(\text{紫色}) = 0.01
\]

**T2T 的困境**：\(\max_j p_\theta(x_j) = 0.28 < \tau\)（例如 \(\tau = 0.5\)），所以 T2T 不做任何操作。尽管模型清楚知道"紫色"是错的（概率仅 0.01），但因为没有单一足够"自信"的替代，T2T 选择按兵不动。

**Remask (low_prob) 的优势**：low_prob 策略只检测 \(p_\theta(x_i^{\text{old}} \mid x_{-i}) < \tau_{\text{remask}}\)，即只关注**旧 token 是否被模型判定为不合理**。\(p_\theta(\text{紫色}) = 0.01 \ll 0.5\)，于是该位置被 remask。在后续迭代中，随着 context 更新，模型可能收敛到一个具体的正确 token。

**形式化**：定义"错误但无法纠正"集合：
\[
\mathcal{S}_{\text{stuck}} = \left\{ i : p_\theta(x_i^{\text{old}} \mid x_{-i}) < \epsilon, \;\text{but}\; \max_j p_\theta(x_j \mid x_{-i}) < \tau \right\}
\]
T2T 对 \(\mathcal{S}_{\text{stuck}}\) 完全无能为力。Remask 可以处理其中任何满足 remask 条件的位置。

### 2.3 Case B：多候选时 T2T 的 greedy 替换风险

即使 T2T 能够触发（某个候选过了阈值），它会立即 commit 到 argmax token。但在分布具有多个几乎等概率的模式时，argmax 不一定是全局最优（考虑到与其他位置的联合一致性）。

Remask 通过将位置退回 `<mask>`，让后续步骤在**更完整的 context**（其他位置已经进一步收敛后）下重新决策，这实质上是一种 **delayed commitment** 策略。

---

## 3. 错误 Context 传播与消除

### 3.1 核心观察

这是支持 Remask 的**最关键**论点。

在迭代去噪过程中，模型对位置 \(i\) 的预测条件于**所有**已 unmask 的 token：

\[
p_\theta(x_i \mid x_{-i}) = p_\theta\!\bigl(x_i \mid \underbrace{x_{\text{correct}}}_{\text{正确}},\; \underbrace{x_{\text{error}}}_{\text{错误}},\; \underbrace{\text{mask}}_{\text{未预测}}\bigr)
\]

**关键区别**在于条件中 \(x_{\text{error}}\) 与 \(\text{mask}\) 对模型行为的影响完全不同：

### 3.2 T2T 的错误传播

考虑 block 内有 \(k\) 个位置存在错误 token。当 T2T 尝试修正位置 \(i\) 时：

\[
\hat{x}_i^{(\text{T2T})} = \arg\max p_\theta(x_i \mid x_{\text{correct}},\; \underbrace{x_{\text{error}}^{(\neq i)}}_{\text{k-1 个错误 token}})
\]

模型是在**包含 \(k-1\) 个错误 token 的 context** 下做预测。错误 token 会对注意力机制产生干扰：
- 错误 token 提供了**错误的语义信号**
- 注意力权重分配到了不相关的 token 上
- 模型的 hidden state 被错误信息"污染"

更严重的是，这构成一个**恶性循环**：
1. 位置 \(i\) 的预测因 \(j\) 处的错误 token 而偏差
2. 位置 \(j\) 的预测因 \(i\) 处的（可能也是错误的）替换而偏差
3. 错误 token 相互强化，系统可能陷入**稳定但错误的不动点**

### 3.3 Remask 的 Context 净化

Remask 将检测到的错误位置设为 `<mask>`：

\[
\hat{x}_i^{(\text{remask})} = \arg\max p_\theta(x_i \mid x_{\text{correct}},\; \underbrace{\text{mask}, \text{mask}, \ldots}_{\text{k 个 mask token}})
\]

这带来了根本性的改善：

**mask token 是"中性"的**——它不提供错误的语义信号，只表示"此处未知"。模型仅根据**确实正确的 token** 来推断被 mask 位置，从而：
- 去除了错误 token 间的相互干扰
- 预测条件更干净（只包含正确信息和中性的 mask）
- 打破了错误传播的恶性循环

### 3.4 条件熵视角

形式化地，设 \(E \subseteq \{1,\ldots,n\}\) 为错误位置集合。对位置 \(i \in E\)：

**T2T**：模型在 context \(C_{\text{T2T}} = x_{\text{correct}} \cup x_{\text{error}}^{E \setminus \{i\}}\) 下预测。其中 \(x_{\text{error}}^{E \setminus \{i\}}\) 本质上是**对抗性噪声**——它不是随机的 mask，而是看起来像合法 token 但语义错误的输入。

**Remask**：模型在 context \(C_{\text{remask}} = x_{\text{correct}} \cup \text{mask}^{E}\) 下预测。mask 是模型在训练中见过的标准输入形式。

可以预期：
\[
H\!\bigl(X_i^{\ast} \mid C_{\text{remask}}\bigr) \leq H\!\bigl(X_i^{\ast} \mid C_{\text{T2T}}\bigr)
\]
即在 remask 的 context 下，模型对正确 token 的条件熵更低（更确定），因为它没有被错误信号干扰。

---

## 4. 训练-推理分布对齐

### 4.1 LLADA 的训练目标

LLADA 的训练过程是标准的 masked diffusion：随机 mask 掉一定比例的 token，让模型从 context（正确 token + mask）中预测被 mask 的内容。即模型学到的条件分布为：

\[
p_\theta(x_i \mid \text{observed}_{\text{correct}}, \text{mask}_{\text{masked}})
\]

**训练时模型从未见过"错误 token"作为 context 的情况**。它只见过两种 context token：
- 真实正确的 token（ground truth）
- `<mask>` token

### 4.2 T2T 引入分布偏移

当 T2T 将一个错误 token 留在序列中，或用另一个（可能也不完全正确的）token 替换时，后续预测的 context 包含了**训练分布中不存在的 noise 类型**——即看起来是真实 token 但语义错误的内容。

这是一种 **distribution shift**：
- 训练分布：\(\{x_{\text{correct}}, \text{mask}\}\)
- T2T 推理分布：\(\{x_{\text{correct}}, x_{\text{error}}, \text{mask}\}\)

模型对 \(x_{\text{error}}\) 没有鲁棒性，因为它在训练中假定所有 non-mask token 都是 ground truth。面对错误 token，模型可能会"认真对待"这些错误信息，做出偏差很大的预测。

### 4.3 Remask 恢复训练分布

Remask 将可疑 token 退回 `<mask>`，恢复到模型训练时见过的分布：

\[
\{x_{\text{correct}}, \text{mask}\} \quad \xleftarrow{\text{remask}} \quad \{x_{\text{correct}}, x_{\text{error}}\}
\]

- 错误 token 是 **unknown unknowns**（模型误以为是正确信息）
- mask 是 **known unknowns**（模型知道需要预测）

Remask 将 unknown unknowns 转化为 known unknowns，使模型的推理回到其训练分布上，从而给出更准确的预测。

---

## 5. 避免过早 Commitment 与探索-利用权衡

### 5.1 T2T 的 Greedy 本质

T2T 是一种 **greedy** 策略：一旦检测到高置信度的不同 token，立即 commit。这个 commit 在后续步骤中：
- 影响其他位置的预测（因为它成为了 context 的一部分）
- 如果 commit 有误，错误会级联传播
- 一旦写入，缺乏有效的"撤回"机制（需要再次触发高置信度的不同 token 才能修正）

### 5.2 Remask 的 Exploration 策略

Remask 是一种更具**探索性**的策略。它不急于 commit，而是：
1. 将可疑位置退回"未决定"状态
2. 等待更多 context 信息收敛
3. 在更完整的信息下重新决策

这类似于**模拟退火**中的"升温"操作：局部提高不确定性，以跳出局部最优，寻找全局更优解。

也类似于**蒙特卡洛树搜索**中的思想：不急于 commit，而是探索更多可能性后再做决定。

### 5.3 形式化：多位置联合最优

在 block 内有多个需要编辑的位置 \(\{i_1, i_2, \ldots, i_k\}\)。真正的目标是联合最优：

\[
\arg\max_{x_{i_1}, \ldots, x_{i_k}} p_\theta(x_{i_1}, \ldots, x_{i_k} \mid x_{\text{correct}})
\]

T2T 用逐位置贪心近似：

\[
\hat{x}_{i_t} = \arg\max p_\theta(x_{i_t} \mid x_{\text{correct}}, \hat{x}_{i_1}, \ldots, \hat{x}_{i_{t-1}}, x_{i_{t+1}}^{\text{old}}, \ldots)
\]

这忽略了位置之间的相互依赖。Remask 通过将多个可疑位置同时 mask，让模型在后续步骤中**同时**考虑这些位置的最优填充，更接近联合优化。

---

## 6. 与连续扩散模型的类比

在连续扩散模型（如 DDPM）中，去噪步骤估计 score function \(\nabla_{x} \log p(x)\)。关键要求是：**输入噪声的分布必须与训练时一致**。如果用错误类型的噪声（如对抗性噪声代替高斯噪声），score 估计会严重偏差。

LLADA 的离散设置中：
- **训练噪声**：`<mask>` token（对应连续扩散的高斯噪声）
- **T2T 引入的噪声**：错误 token（对应对抗性扰动）
- **Remask 恢复的噪声**：`<mask>` token（回到训练噪声类型）

T2T 本质上是在用对抗噪声代替训练噪声，导致去噪轨迹偏离了模型学到的最优路径。Remask 保持噪声类型一致，确保去噪过程沿着正确的轨迹运行。

---

## 7. 总结

| 维度 | T2T (Token Edit) | T2M (Remask) |
|------|-----------------|--------------|
| 错误检测 | 需同时满足"新 token 高置信"+"与旧不同"，耦合了检测与替代 | 仅检测旧 token 不合理，解耦了检测与替代 |
| 多模态分布 | 无法处理（无单一高置信候选时失效） | 可处理（只需判定旧 token 不对，新 token 由后续迭代决定） |
| 错误 context | 错误 token 保留在序列中，污染后续预测 | 错误 token 被替换为中性的 mask，净化 context |
| 训练-推理对齐 | 引入 OOD 的错误 token context，模型未见过 | 恢复训练分布（correct + mask） |
| Commitment | Greedy，立即 commit，错误级联 | 延迟 commit，探索更多可能性 |
| 联合优化 | 逐位置贪心，忽略位置间依赖 | 多位置同时 mask，更接近联合优化 |
| 类比连续扩散 | 用对抗噪声替代训练噪声，score 估计偏差 | 保持训练噪声类型，score 估计无偏 |

**核心结论**：Remask 的优势源于一个统一的直觉——**将不确定的位置退回到模型训练时见过的"未知"状态（mask），而不是保留或替换为可能错误的 token**。这同时实现了 context 净化、分布对齐、和延迟 commitment，从多个维度改善了迭代去噪的质量。

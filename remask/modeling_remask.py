"""LLaDA2.1 with T2M remasking — only overrides _token_edit_step.

Three remask strategies, selected by self.remask_strategy:

  "low_prob"   (策略1): 当前 logits 对旧 token 的预测概率低 → remask
               P_new(old_token) < remask_threshold → remask
               即：模型重新看了一遍上下文，觉得这个位置不该是这个 token

  "t2t_remask" (策略2): 和 T2T 同条件 (新 token prob 高 + 和旧 token 不同) → remask
               x0_p > remask_threshold AND x0 != old_token → remask
               即：模型很自信地想换一个不同的 token，说明旧的不对

  "logit_diff" (策略3): 新旧两步 logits 归一化后对该 token 的概率差异大 → remask
               |P_new(old_token) - P_prev(old_token)| > remask_threshold → remask
               即：模型对这个位置的看法在两步之间变化剧烈，说明不稳定

防死循环: 每个位置最多被 remask max_remask_per_pos 次 (默认 3)
"""

import torch
import torch.nn.functional as F

from .modeling_llada2_moe import LLaDA2MoeModelLM


REMASK_STRATEGIES = ("low_prob", "t2t_remask", "logit_diff")

STRATEGY_DEFAULTS = {
    #                       remask_threshold
    "low_prob":             0.5,    # P_new(old_token) < 0.5 → remask
    "t2t_remask":           0.9,    # P(new_token) > 0.9 且 new≠old → remask
    "logit_diff":           0.3,    # |ΔP(old_token)| > 0.3 → remask
}


class LLaDA2MoeRemaskLM(LLaDA2MoeModelLM):
    """
    Args:
        config:              LLaDA2MoeConfig (from pretrained)
        remask_strategy:     "low_prob" | "t2t_remask" | "logit_diff"
        remask_threshold:    各策略的阈值, None 则用 STRATEGY_DEFAULTS 默认值
        max_remask_per_pos:  每个位置最多被 remask 的次数 (防死循环)
    """

    def __init__(self, config, remask_strategy="low_prob",
                 remask_threshold=None, max_remask_per_pos=5):
        super().__init__(config)
        assert remask_strategy in REMASK_STRATEGIES, \
            f"remask_strategy must be one of {REMASK_STRATEGIES}, got {remask_strategy!r}"
        self.remask_strategy = remask_strategy
        self.remask_threshold = (
            remask_threshold if remask_threshold is not None
            else STRATEGY_DEFAULTS[remask_strategy]
        )
        self.max_remask_per_pos = max_remask_per_pos
        self._prev_logits = None
        self._remask_counts = None

    def _get_editable_mask(self, active_block_mask, prompt_mask_in_block):
        non_mask = ~active_block_mask
        non_prompt = ~prompt_mask_in_block
        return non_mask & non_prompt[None, :]

    # ── 策略 1: P_new(old_token) 低 → remask ──────────────────────────

    def _remask_low_prob(self, active_logits, old_block_tokens, editable, threshold):
        """当前 logits softmax 后，旧 token 的概率低于 threshold 就 remask."""
        probs = F.softmax(active_logits[0].float(), dim=-1)
        old_tok_prob = probs.gather(1, old_block_tokens[0].unsqueeze(-1)).squeeze(-1)
        return (old_tok_prob < threshold) & editable[0]

    # ── 策略 2: 和 T2T 同条件，但 remask 而非替换 ─────────────────────

    def _remask_t2t(self, x0, x0_p, old_block_tokens, editable, threshold):
        """新 token prob 高 + 和旧 token 不同 → 旧位置 remask."""
        high_conf = (x0_p[0] > threshold) & editable[0]
        changed = x0[0] != old_block_tokens[0]
        return high_conf & changed

    # ── 策略 3: 新旧 logits 归一化概率差异大 → remask ─────────────────

    def _remask_logit_diff(self, active_logits, old_block_tokens, editable, threshold):
        """两步之间，该 token 的预测概率变化超过 threshold → remask."""
        probs_new = F.softmax(active_logits[0].float(), dim=-1)
        p_new = probs_new.gather(1, old_block_tokens[0].unsqueeze(-1)).squeeze(-1)

        if self._prev_logits is None or self._prev_logits.shape != active_logits.shape:
            self._prev_logits = active_logits.detach().clone()
            return torch.zeros_like(editable[0])

        probs_old = F.softmax(self._prev_logits[0].float(), dim=-1)
        p_old = probs_old.gather(1, old_block_tokens[0].unsqueeze(-1)).squeeze(-1)

        self._prev_logits = active_logits.detach().clone()
        return ((p_old - p_new).abs() > threshold) & editable[0]

    # ── 总入口 ─────────────────────────────────────────────────────────

    def _token_edit_step(
        self, cur_x, x0, x0_p, active_logits, mask_transfer_index,
        old_block_tokens, active_block_mask,
        prompt_mask_in_block, editing_threshold,
        block_length, mask_id,
    ):
        # M2T: 正常把高置信 mask 位填上新 token（和原版一样）
        if mask_transfer_index.any():
            cur_x[:, -block_length:][mask_transfer_index] = x0[mask_transfer_index]

        editable = self._get_editable_mask(active_block_mask, prompt_mask_in_block)
        if not editable.any():
            self._remask_counts = None
            self._prev_logits = None
            return True

        # 初始化 / 重置 per-position 计数器（block_length 变了说明换 block 了）
        if (self._remask_counts is None
                or self._remask_counts.shape[0] != block_length):
            self._remask_counts = torch.zeros(
                block_length, dtype=torch.long, device=cur_x.device)

        thr = self.remask_threshold

        if self.remask_strategy == "low_prob":
            remask = self._remask_low_prob(
                active_logits, old_block_tokens, editable, thr)
        elif self.remask_strategy == "t2t_remask":
            remask = self._remask_t2t(
                x0, x0_p, old_block_tokens, editable, thr)
        elif self.remask_strategy == "logit_diff":
            remask = self._remask_logit_diff(
                active_logits, old_block_tokens, editable, thr)
        else:
            raise ValueError(f"Unknown remask_strategy: {self.remask_strategy}")

        # 每个位置最多 remask max_remask_per_pos 次
        remask = remask & (self._remask_counts < self.max_remask_per_pos)

        if remask.any():
            cur_x[0, -block_length:][remask] = mask_id
            self._remask_counts[remask] += 1

        return not remask.any()

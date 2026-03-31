"""LLaDA2.1 with T2M remasking — only overrides _token_edit_step.

Three remask strategies, selected by self.remask_strategy:

  "low_prob"   (策略1): 当前 logits 对旧 token 的预测概率低 → remask
               P_new(old_token) < remask_threshold → remask
               即：模型重新看了一遍上下文，觉得这个位置不该是这个 token

  "t2t_remask" (策略2): 和 T2T 同条件 (新 token prob 高 + 和旧 token 不同) → remask
               x0_p > remask_threshold AND x0 != old_token → remask
               即：模型很自信地想换一个不同的 token，说明旧的不对

  "logit_diff" (策略3): 该 token 的预测概率在两步间下降超过阈值 → remask
               P_prev(old_token) - P_new(old_token) > remask_threshold → remask
               即：模型对这个 token 的置信度下降了，说明它与演化的上下文不兼容

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
    "logit_diff":           0.3,    # P_prev - P_new > 0.3 → remask
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
                 remask_threshold=None, max_remask_per_pos=5,
                 max_remask_ratio=0.25):
        super().__init__(config)
        assert remask_strategy in REMASK_STRATEGIES, \
            f"remask_strategy must be one of {REMASK_STRATEGIES}, got {remask_strategy!r}"
        self.remask_strategy = remask_strategy
        self.remask_threshold = (
            remask_threshold if remask_threshold is not None
            else STRATEGY_DEFAULTS[remask_strategy]
        )
        self.max_remask_per_pos = max_remask_per_pos
        self.max_remask_ratio = max_remask_ratio
        self._prev_logits = None
        self._remask_counts = None
        self._batch_prev_logits = {}
        self._batch_remask_counts = {}

    def _get_editable_mask(self, active_block_mask, prompt_mask_in_block):
        non_mask = ~active_block_mask
        non_prompt = ~prompt_mask_in_block
        return non_mask & non_prompt[None, :]

    # ── 策略 1: P_new(old_token) 低 → remask ──────────────────────────

    def _remask_low_prob(self, active_logits, old_block_tokens, editable, threshold):
        """当前 logits softmax 后，旧 token 的概率低于 threshold 就 remask.
        Returns (remask_mask, confidence_scores) — scores 越低越该被 remask.
        """
        probs = F.softmax(active_logits[0].float(), dim=-1)
        old_tok_prob = probs.gather(1, old_block_tokens[0].unsqueeze(-1)).squeeze(-1)
        return (old_tok_prob < threshold) & editable[0], old_tok_prob

    # ── 策略 2: 和 T2T 同条件，但 remask 而非替换 ─────────────────────

    def _remask_t2t(self, x0, x0_p, old_block_tokens, editable, threshold):
        """新 token prob 高 + 和旧 token 不同 → 旧位置 remask.
        Returns (remask_mask, confidence_scores) — 用 1-x0_p 作为 score,
        越低表示新 token 越自信, 旧 token 越该被 remask.
        """
        high_conf = (x0_p[0] > threshold) & editable[0]
        changed = x0[0] != old_block_tokens[0]
        return high_conf & changed, 1.0 - x0_p[0]

    # ── 策略 3: 新旧 logits 归一化概率差异大 → remask ─────────────────

    def _remask_logit_diff(self, active_logits, old_block_tokens, editable, threshold):
        """两步之间，该 token 的预测概率下降超过 threshold → remask.
        只在置信度下降（p_old > p_new）时触发，置信度上升说明 token 被认可。
        Returns (remask_mask, confidence_scores) — drop 越大越该被 remask,
        所以 score = -drop (越低越该 remask).
        """
        probs_new = F.softmax(active_logits[0].float(), dim=-1)
        p_new = probs_new.gather(1, old_block_tokens[0].unsqueeze(-1)).squeeze(-1)

        bidx = getattr(self, '_batch_sample_idx', None)
        if bidx is not None:
            prev = self._batch_prev_logits.get(bidx)
        else:
            prev = self._prev_logits

        if prev is None or prev.shape != active_logits.shape:
            if bidx is not None:
                self._batch_prev_logits[bidx] = active_logits.detach().clone()
            else:
                self._prev_logits = active_logits.detach().clone()
            zeros = torch.zeros_like(editable[0])
            return zeros, torch.zeros(editable.shape[1], device=editable.device)

        probs_old = F.softmax(prev[0].float(), dim=-1)
        p_old = probs_old.gather(1, old_block_tokens[0].unsqueeze(-1)).squeeze(-1)

        if bidx is not None:
            self._batch_prev_logits[bidx] = active_logits.detach().clone()
        else:
            self._prev_logits = active_logits.detach().clone()
        drop = p_old - p_new
        return (drop > threshold) & editable[0], -drop

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
            self._reset_remask_state()
            return True

        bidx = getattr(self, '_batch_sample_idx', None)
        remask_counts = self._get_remask_counts(bidx, block_length, cur_x.device)

        thr = self.remask_threshold

        if self.remask_strategy == "low_prob":
            remask, scores = self._remask_low_prob(
                active_logits, old_block_tokens, editable, thr)
        elif self.remask_strategy == "t2t_remask":
            remask, scores = self._remask_t2t(
                x0, x0_p, old_block_tokens, editable, thr)
        elif self.remask_strategy == "logit_diff":
            remask, scores = self._remask_logit_diff(
                active_logits, old_block_tokens, editable, thr)
        else:
            raise ValueError(f"Unknown remask_strategy: {self.remask_strategy}")

        remask = remask & (remask_counts < self.max_remask_per_pos)

        if remask.any() and self.max_remask_ratio < 1.0:
            n_editable = editable.sum().item()
            max_allowed = max(1, int(n_editable * self.max_remask_ratio))
            n_remask = remask.sum().item()
            if n_remask > max_allowed:
                candidate_scores = torch.where(remask, scores, torch.inf)
                _, topk_idx = candidate_scores.topk(max_allowed, largest=False)
                new_remask = torch.zeros_like(remask)
                new_remask[topk_idx] = True
                remask = new_remask

        if remask.any():
            cur_x[0, -block_length:][remask] = mask_id
            remask_counts[remask] += 1
            if hasattr(self, '_gen_remask_total'):
                self._gen_remask_total += remask.sum().item()

        return not remask.any()

    def _get_remask_counts(self, batch_idx, block_length, device):
        """Get or initialize per-position remask counter (batch-aware)."""
        if batch_idx is not None:
            counts = self._batch_remask_counts.get(batch_idx)
            if counts is None or counts.shape[0] != block_length:
                counts = torch.zeros(block_length, dtype=torch.long, device=device)
                self._batch_remask_counts[batch_idx] = counts
            return counts
        else:
            if (self._remask_counts is None
                    or self._remask_counts.shape[0] != block_length):
                self._remask_counts = torch.zeros(
                    block_length, dtype=torch.long, device=device)
            return self._remask_counts

    def _reset_remask_state(self):
        bidx = getattr(self, '_batch_sample_idx', None)
        if bidx is not None:
            self._batch_remask_counts.pop(bidx, None)
            self._batch_prev_logits.pop(bidx, None)
        else:
            self._remask_counts = None
            self._prev_logits = None

    def generate(self, *args, **kwargs):
        self._gen_remask_total = 0
        result = super().generate(*args, **kwargs)
        if hasattr(self, '_gen_stats'):
            self._gen_stats['remask_total'] = self._gen_remask_total
        return result

    def generate_batch(self, inputs_list, **kwargs):
        """Override to reset batch state before and after."""
        self._batch_prev_logits = {}
        self._batch_remask_counts = {}
        self._gen_remask_total = 0
        try:
            result = super().generate_batch(inputs_list, **kwargs)
            if hasattr(self, '_gen_stats'):
                self._gen_stats['remask_total'] = self._gen_remask_total
            return result
        finally:
            self._batch_prev_logits = {}
            self._batch_remask_counts = {}

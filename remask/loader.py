"""Model loading helpers."""

import os
import sys

import remask.env  # noqa: F401 — sets HF_HOME, HF_ENDPOINT, cache dirs

import torch
from transformers import AutoTokenizer

MODEL_PATH = "/vepfs-mlp2/c20250506/251105017/yaolin/LLaDA2.1-mini"
MASK_ID = 156895


def _get_tokenizer_and_mask(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    mask_id = tokenizer.convert_tokens_to_ids("<|mask|>")
    if mask_id is None or mask_id == tokenizer.unk_token_id:
        mask_id = MASK_ID
    return tokenizer, mask_id


def load_original_model(model_path=None, device_map="auto"):
    """Load LLaDA2.1-mini with original T2T generate()."""
    from .modeling_llada2_moe import LLaDA2MoeModelLM, LLaDA2MoeConfig

    if model_path is None:
        model_path = MODEL_PATH
    config = LLaDA2MoeConfig.from_pretrained(model_path)
    model = LLaDA2MoeModelLM.from_pretrained(
        model_path, config=config,
        dtype=torch.bfloat16, device_map=device_map,
    ).eval()
    tokenizer, mask_id = _get_tokenizer_and_mask(model_path)
    return model, tokenizer, mask_id


def load_remask_model(model_path=None, device_map="auto",
                      strategy="low_prob", remask_threshold=None,
                      max_remask_per_pos=3, max_remask_ratio=0.25):
    """Load LLaDA2.1-mini with T2M remask generate().

    strategy:            "low_prob" | "t2t_remask" | "logit_diff"
    remask_threshold:    override default threshold for the chosen strategy
    max_remask_per_pos:  每个位置最多被 remask 几次 (防死循环)
    max_remask_ratio:    每步最多 remask 多少比例的 editable token (防过度 remask)
    """
    from .modeling_remask import LLaDA2MoeRemaskLM
    from .modeling_llada2_moe import LLaDA2MoeConfig

    if model_path is None:
        model_path = MODEL_PATH
    config = LLaDA2MoeConfig.from_pretrained(model_path)
    model = LLaDA2MoeRemaskLM.from_pretrained(
        model_path, config=config,
        dtype=torch.bfloat16, device_map=device_map,
    ).eval()
    model.remask_strategy = strategy
    if remask_threshold is not None:
        model.remask_threshold = remask_threshold
    model.max_remask_per_pos = max_remask_per_pos
    model.max_remask_ratio = max_remask_ratio
    tokenizer, mask_id = _get_tokenizer_and_mask(model_path)
    return model, tokenizer, mask_id

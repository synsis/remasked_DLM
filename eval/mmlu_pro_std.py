"""MMLU-Pro with 5-shot CoT (LLaMA 3.1 / lm-evaluation-harness standard).

Uses per-category 5-shot examples from the validation split, matching
the exact prompt format from lm-eval-harness.

python -m eval.mmlu_pro_std --mode remask --strategy low_prob --remask_threshold 0.3
"""

import argparse
import os
from collections import defaultdict

import remask.env  # noqa: F401

from datasets import load_dataset

from remask import load_remask_model, load_original_model
from remask.utils import extract_choice_answer
from eval.common import add_parallel_args, shard_dataset, run_eval

LETTERS = "ABCDEFGHIJ"
N_FEWSHOT = 5

_fewshot_by_category = {}


def _fmt_choices(opts):
    return "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts) if i < len(LETTERS))


def _format_example(ex, include_answer=True):
    """Format one MMLU-Pro example matching lm-eval-harness standard."""
    prompt = "Question:\n"
    prompt += ex["question"] + "\n"
    prompt += "Options:\n"
    prompt += _fmt_choices(ex["options"]) + "\n"
    if include_answer:
        cot = ex["cot_content"].replace(
            "A: Let's think step by step.",
            "Answer: Let's think step by step."
        )
        prompt += cot + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def _load_fewshot(hf_token):
    global _fewshot_by_category
    if _fewshot_by_category:
        return
    val_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation", token=hf_token)
    by_cat = defaultdict(list)
    for ex in val_ds:
        cat = ex.get("category", "unknown")
        if len(by_cat[cat]) < N_FEWSHOT:
            by_cat[cat].append(_format_example(ex, include_answer=True))
    _fewshot_by_category = dict(by_cat)
    print(f"Loaded MMLU-Pro 5-shot CoT for {len(_fewshot_by_category)} categories")


def run_tag(args):
    if args.mode == "original":
        return "original"
    return f"remask_{args.strategy}_{args.remask_threshold}"


def format_prompt(ex, tokenizer):
    from remask.utils import format_chat_prompt
    cat = ex.get("category", "unknown")
    shots = _fewshot_by_category.get(cat, [])
    prefix = "".join(shots)
    test_q = _format_example(ex, include_answer=False)
    return format_chat_prompt(prefix + test_q, tokenizer)


def evaluate(resp, ex):
    gold_idx = ex["answer_index"]
    gold = LETTERS[gold_idx] if isinstance(gold_idx, int) else ex.get("answer", "")
    pred = extract_choice_answer(resp)
    return dict(question=ex["question"], category=ex.get("category", "unknown"),
                gold=gold, predicted=pred, correct=(pred == gold))


def run(args):
    hf_token = os.environ.get("HF_TOKEN", None)
    _load_fewshot(hf_token)
    tag = run_tag(args)

    if args.mode == "original":
        model, tokenizer, mask_id = load_original_model(args.model_path)
    else:
        model, tokenizer, mask_id = load_remask_model(
            args.model_path, strategy=args.strategy,
            remask_threshold=args.remask_threshold,
            max_remask_per_pos=getattr(args, "max_remask_per_pos", 3),
            max_remask_ratio=getattr(args, "max_remask_ratio", 0.25))

    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test", token=hf_token)
    print(f"MMLU-Pro: {len(dataset)} problems (5-shot CoT)")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = shard_dataset(dataset, args)

    run_eval(model, tokenizer, mask_id, list(dataset), args, tag, "mmlu_pro",
             format_prompt, evaluate)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/mmlu_pro")
    p.add_argument("--gen_length", type=int, default=2048)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_remask_per_pos", type=int, default=3)
    p.add_argument("--max_remask_ratio", type=float, default=0.25)
    add_parallel_args(p)
    run(p.parse_args())

"""BBH with 3-shot CoT (LLaMA 3.1 / lm-evaluation-harness standard).

Uses the original BIG-Bench-Hard CoT prompts from Suzgun et al.
(downloaded and cached from github.com/suzgunmirac/BIG-Bench-Hard).

python -m eval.bbh_std --mode remask --strategy low_prob --remask_threshold 0.3
"""

import argparse
import os
import re
import string
import unicodedata
import urllib.request

import remask.env  # noqa: F401

from datasets import load_dataset, concatenate_datasets

from remask import load_remask_model, load_original_model
from remask.utils import compute_em
from eval.common import add_parallel_args, shard_dataset, run_eval

BBH_COT_BASE = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/cot-prompts"
BBH_COT_CACHE_DIR = os.path.expanduser("~/.cache/bbh_cot_prompts")

BBH_CONFIGS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate",
    "object_counting", "penguins_in_a_table",
    "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks", "sports_understanding",
    "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting",
]

_cot_prompts = {}


def _fetch_cot_prompt(subtask):
    cache_path = os.path.join(BBH_COT_CACHE_DIR, f"{subtask}.txt")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return f.read()
    os.makedirs(BBH_COT_CACHE_DIR, exist_ok=True)
    url = f"{BBH_COT_BASE}/{subtask}.txt"
    text = urllib.request.urlopen(url).read().decode("utf-8")
    with open(cache_path, "w") as f:
        f.write(text)
    return text


def _parse_cot_prompt(raw):
    parts = raw.split("-----\n", 1)
    body = parts[1] if len(parts) > 1 else raw
    return body.strip()


def _load_cot_prompts():
    global _cot_prompts
    if _cot_prompts:
        return
    for cfg in BBH_CONFIGS:
        raw = _fetch_cot_prompt(cfg)
        _cot_prompts[cfg] = _parse_cot_prompt(raw)
    print(f"Loaded {len(_cot_prompts)} BBH CoT prompts")


def _norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def bbh_correct(resp, target):
    if compute_em(resp, target) >= 1.0:
        return True
    rn, tn = _norm(resp), _norm(target)
    if not tn:
        return False
    return tn in rn or rn == tn


def load_bbh_with_subtasks():
    errors = []
    for repo in ["lukaemon/bbh", "maveriq/bigbenchhard"]:
        split = "test" if "lukaemon" in repo else "train"
        try:
            parts = []
            for cfg in BBH_CONFIGS:
                ds = load_dataset(repo, cfg, split=split)
                ds = ds.add_column("subtask", [cfg] * len(ds))
                parts.append(ds)
            return concatenate_datasets(parts)
        except Exception as e:
            errors.append(f"{repo}: {e}")
    raise RuntimeError("BBH load failed: " + "; ".join(errors))


def run_tag(args):
    if args.mode == "original":
        return "original"
    return f"remask_{args.strategy}_{args.remask_threshold}"


def format_prompt(ex, tokenizer):
    from remask.utils import format_chat_prompt
    subtask = ex["subtask"]
    cot_prefix = _cot_prompts[subtask]
    user_msg = f"{cot_prefix}\n\nQ: {ex['input']}\nA: Let's think step by step.\n"
    return format_chat_prompt(user_msg, tokenizer)


def evaluate(resp, ex):
    target = ex["target"]
    m = re.search(r"[Ss]o the answer is (.+?)\.?\s*$", resp, re.MULTILINE)
    if m:
        pred = m.group(1).strip()
    else:
        m = re.search(r"the answer is (.+?)\.?\s*$", resp, re.IGNORECASE | re.MULTILINE)
        pred = m.group(1).strip() if m else resp.strip().split("\n")[-1].strip()
    ok = bbh_correct(pred, target)
    return dict(input=ex["input"], subtask=ex["subtask"], target=target,
                predicted=pred, correct=ok)


def run(args):
    _load_cot_prompts()
    tag = run_tag(args)
    if args.mode == "original":
        model, tokenizer, mask_id = load_original_model(args.model_path)
    else:
        model, tokenizer, mask_id = load_remask_model(
            args.model_path, strategy=args.strategy,
            remask_threshold=args.remask_threshold,
            max_remask_per_pos=getattr(args, "max_remask_per_pos", 3),
            max_remask_ratio=getattr(args, "max_remask_ratio", 0.25))

    dataset = load_bbh_with_subtasks()
    print(f"BBH: {len(dataset)} items (3-shot CoT)")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = shard_dataset(dataset, args)

    run_eval(model, tokenizer, mask_id, list(dataset), args, tag, "bbh",
             format_prompt, evaluate)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/bbh")
    p.add_argument("--gen_length", type=int, default=1024)
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

"""
python -m eval.bbh --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.bbh --mode original --batch_size 4
"""

import argparse
import remask.env  # noqa: F401
import re
import string
import unicodedata

from datasets import concatenate_datasets, get_dataset_config_names, load_dataset

from remask import load_remask_model, load_original_model
from remask.utils import compute_em
from eval.common import add_parallel_args, shard_dataset, run_eval


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


def load_bbh():
    try:
        configs = get_dataset_config_names("lukaemon/bbh")
        parts = [load_dataset("lukaemon/bbh", c, split="test") for c in configs]
        return concatenate_datasets(parts)
    except Exception as e1:
        try:
            return load_dataset("maveriq/bigbenchhard", split="train")
        except Exception as e2:
            raise RuntimeError(
                f"BBH load failed (lukaemon/bbh: {e1}; maveriq/bigbenchhard: {e2})"
            ) from e2


def run_tag(args):
    if args.mode == "original":
        return "original"
    return f"remask_{args.strategy}_{args.remask_threshold}"


def format_prompt(ex, tokenizer):
    from remask.utils import format_chat_prompt
    user = f"Follow the given examples and answer the question.\n\n{ex['input']}\nAnswer:"
    return format_chat_prompt(user, tokenizer)


def evaluate(resp, ex):
    target = ex["target"]
    ok = bbh_correct(resp, target)
    return dict(input=ex["input"], target=target, correct=ok)


def run(args):
    tag = run_tag(args)
    if args.mode == "original":
        model, tokenizer, mask_id = load_original_model(args.model_path)
    else:
        model, tokenizer, mask_id = load_remask_model(
            args.model_path, strategy=args.strategy,
            remask_threshold=args.remask_threshold,
            max_remask_per_pos=getattr(args, "max_remask_per_pos", 3),
            max_remask_ratio=getattr(args, "max_remask_ratio", 0.25))

    dataset = load_bbh()
    print(f"BBH: {len(dataset)} items")
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
    p.add_argument("--gen_length", type=int, default=16384)
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

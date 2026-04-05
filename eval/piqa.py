"""
python -m eval.piqa --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.piqa --mode original --batch_size 128
"""

import argparse
import remask.env  # noqa: F401

from datasets import load_dataset

from remask import load_remask_model, load_original_model
from remask.utils import extract_choice_answer
from eval.common import add_parallel_args, shard_dataset, run_eval


def load_piqa():
    for repo, split in [("piqa", "validation"), ("ybisk/piqa", "validation"),
                         ("lmms-lab/piqa", "test")]:
        try:
            return load_dataset(repo, split=split)
        except Exception:
            continue
    raise RuntimeError("Cannot load PIQA dataset")


def run_tag(args):
    if args.mode == "original":
        return "original"
    return f"remask_{args.strategy}_{args.remask_threshold}"


def format_prompt(ex, tokenizer):
    from remask.utils import format_chat_prompt
    goal = ex["goal"]
    sol1, sol2 = ex["sol1"], ex["sol2"]
    user = (
        "Choose the better solution to achieve the goal.\n\n"
        f"Goal: {goal}\n\n"
        f"A. {sol1}\nB. {sol2}\n\n"
        "The answer is"
    )
    return format_chat_prompt(user, tokenizer)


def evaluate(resp, ex):
    gold = "A" if int(ex["label"]) == 0 else "B"
    pred = extract_choice_answer(resp, 2)
    ok = pred == gold
    return dict(goal=ex["goal"], gold=gold, predicted=pred, correct=ok)


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

    dataset = load_piqa()
    print(f"PIQA: {len(dataset)} items")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = shard_dataset(dataset, args)

    run_eval(model, tokenizer, mask_id, list(dataset), args, tag, "piqa",
             format_prompt, evaluate)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/piqa")
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

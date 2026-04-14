"""
python -m eval.aime2025 --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.aime2025 --mode original --batch_size 4
"""

import argparse
import json
import os
import remask.env  # noqa: F401

from remask import load_remask_model, load_original_model
from remask.utils import extract_math_answer, normalize_numeric
from eval.common import add_parallel_args, run_eval

AIME_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your answer inside \\boxed{{}}.\n\n"
    "{problem}\n\n"
    "Remember to put your answer inside \\boxed{{}}."
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATA = os.path.join(_ROOT, "data", "aime2025.json")


def run_tag(args):
    if args.mode == "original":
        return "original"
    return f"remask_{args.strategy}_{args.remask_threshold}"


def load_problems(path):
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("aime2025.json must be a JSON list of {problem, answer}")
    return data


def format_prompt(ex, tokenizer):
    from remask.utils import format_chat_prompt
    return format_chat_prompt(AIME_PROMPT.format(problem=ex["problem"]), tokenizer)


def evaluate(resp, ex):
    gold = str(ex["answer"]).strip()
    pred = extract_math_answer(resp)
    ok = normalize_numeric(pred) == normalize_numeric(gold)
    return dict(problem=ex["problem"], gold=gold, predicted=pred, correct=ok)


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

    problems = load_problems(_DEFAULT_DATA)
    print(f"AIME 2025: {len(problems)} problems from {_DEFAULT_DATA}")
    if args.max_samples:
        problems = problems[: min(args.max_samples, len(problems))]
    if args.num_shards > 1:
        problems = [p for i, p in enumerate(problems) if i % args.num_shards == args.shard_id]
        print(f"  shard {args.shard_id}/{args.num_shards}: {len(problems)} problems")

    run_eval(model, tokenizer, mask_id, problems, args, tag, "aime2025",
             format_prompt, evaluate)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/aime2025")
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

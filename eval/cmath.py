"""
python -m eval.cmath --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.cmath --mode original
python -m eval.cmath --mode original --batch_size 4
"""

import argparse
import remask.env  # noqa: F401

from datasets import load_dataset

from remask import load_remask_model, load_original_model
from remask.utils import extract_math_answer, normalize_numeric
from eval.common import add_parallel_args, shard_dataset, run_eval

CMATH_PROMPT = (
    "请逐步解决以下数学问题。\n\n"
    "问题：{question}\n\n"
    "请一步一步思考，最后一行请用「答案是X」的格式给出最终数字答案。"
)


def get_gold(ex):
    raw = ex.get("golden")
    if raw is None:
        raw = ex.get("answer")
    return str(raw).strip() if raw is not None else ""


def run_tag(args):
    if args.mode == "original":
        return "original"
    c = getattr(args, 'max_remask_per_pos', 3)
    r = getattr(args, 'max_remask_ratio', 0.25)
    return f"remask_{args.strategy}_{args.remask_threshold}_c{c}_r{r}"


def load_cmath():
    last_err = None
    for path in ("weitianwen/cmath", "CMATH/CMATH"):
        try:
            return load_dataset(path, split="test")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load cmath dataset: {last_err}")


def format_prompt(ex, tokenizer):
    from remask.utils import format_chat_prompt
    return format_chat_prompt(
        CMATH_PROMPT.format(question=ex["question"]), tokenizer
    )


def evaluate(resp, ex):
    gold = get_gold(ex)
    pred = extract_math_answer(resp)
    ok = normalize_numeric(pred) == normalize_numeric(gold)
    return dict(question=ex["question"], gold=gold, predicted=pred, correct=ok)


def run(args):
    tag = run_tag(args)
    if args.mode == "original":
        model, tokenizer, mask_id = load_original_model(args.model_path)
    else:
        model, tokenizer, mask_id = load_remask_model(
            args.model_path, strategy=args.strategy,
            remask_threshold=args.remask_threshold,
            max_remask_per_pos=getattr(args, 'max_remask_per_pos', 3),
            max_remask_ratio=getattr(args, 'max_remask_ratio', 0.25))

    dataset = load_cmath()
    print(f"cmath: {len(dataset)} problems")
    if args.max_samples:
        if args.sample_seed is not None:
            import random
            random.seed(args.sample_seed)
            indices = sorted(random.sample(range(len(dataset)), min(args.max_samples, len(dataset))))
            dataset = dataset.select(indices)
            print(f"  random sampled {len(dataset)} with seed={args.sample_seed}")
        else:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = shard_dataset(dataset, args)

    run_eval(model, tokenizer, mask_id, list(dataset), args, tag, "cmath",
             format_prompt, evaluate)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/cmath")
    p.add_argument("--gen_length", type=int, default=16384)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_remask_per_pos", type=int, default=3)
    p.add_argument("--max_remask_ratio", type=float, default=0.25)
    p.add_argument("--sample_seed", type=int, default=None,
                   help="If set, randomly sample max_samples from dataset with this seed")
    add_parallel_args(p)
    run(p.parse_args())

"""
python -m eval.mmlu_pro --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.mmlu_pro --mode original --batch_size 4
"""

import argparse
import remask.env  # noqa: F401

from datasets import load_dataset

from remask import load_remask_model, load_original_model
from remask.utils import extract_choice_answer
from eval.common import add_parallel_args, shard_dataset, run_eval

LETTERS = "ABCDEFGHIJ"

PROMPT_TPL = (
    "The following is a multiple choice question. "
    "Think step by step and then output the answer in the format "
    '"The answer is (X)" where X is the correct letter.\n\n'
    "Question: {question}\n\n{choices}\n\nAnswer:"
)


def fmt_choices(opts):
    return "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts) if i < len(LETTERS))


def run_tag(args):
    if args.mode == "original":
        return "original"
    return f"remask_{args.strategy}_{args.remask_threshold}"


def format_prompt(ex, tokenizer):
    from remask.utils import format_chat_prompt
    return format_chat_prompt(
        PROMPT_TPL.format(question=ex["question"], choices=fmt_choices(ex["options"])),
        tokenizer)


def evaluate(resp, ex):
    gold_idx = ex["answer_index"]
    gold = LETTERS[gold_idx] if isinstance(gold_idx, int) else ex.get("answer", "")
    pred = extract_choice_answer(resp)
    return dict(question=ex["question"], category=ex.get("category", "unknown"),
                gold=gold, predicted=pred, correct=(pred == gold))


def run(args):
    tag = run_tag(args)
    if args.mode == "original":
        model, tokenizer, mask_id = load_original_model(args.model_path)
    else:
        model, tokenizer, mask_id = load_remask_model(
            args.model_path, strategy=args.strategy,
            remask_threshold=args.remask_threshold)

    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    print(f"MMLU-Pro: {len(dataset)} problems")
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
    p.add_argument("--gen_length", type=int, default=16384)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    add_parallel_args(p)
    run(p.parse_args())

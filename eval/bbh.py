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


def load_bbh():
    errors = []
    for repo in ["lukaemon/bbh", "maveriq/bigbenchhard"]:
        try:
            configs = get_dataset_config_names(repo)
        except Exception:
            configs = BBH_CONFIGS
        split = "test" if "lukaemon" in repo else "train"
        try:
            parts = [load_dataset(repo, c, split=split) for c in configs]
            return concatenate_datasets(parts)
        except Exception as e:
            errors.append(f"{repo}: {e}")
    raise RuntimeError(f"BBH load failed: " + "; ".join(errors))


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

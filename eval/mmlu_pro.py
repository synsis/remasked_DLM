"""
python -m eval.mmlu_pro --mode remask --strategy low_prob --remask_threshold 0.1
"""

import argparse
import json
import os
import remask.env  # noqa: F401
import time

import torch
from datasets import load_dataset
from tqdm import tqdm

from remask import load_remask_model, load_original_model
from remask.utils import format_chat_prompt, tokenize_prompt, extract_choice_answer

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

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{tag}_results.jsonl")
    correct = total = 0
    results = []
    cat_stats = {}
    t0 = time.time()

    for i, ex in enumerate(tqdm(dataset, desc=f"MMLU-Pro [{tag}]")):
        gold_idx = ex["answer_index"]
        gold = LETTERS[gold_idx] if isinstance(gold_idx, int) else ex.get("answer", "")
        cat = ex.get("category", "unknown")

        prompt = format_chat_prompt(
            PROMPT_TPL.format(question=ex["question"], choices=fmt_choices(ex["options"])),
            tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(
            inputs=ids, gen_length=args.gen_length, block_length=args.block_length,
            steps=args.steps, threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            temperature=args.temperature, mask_id=mask_id, eos_early_stop=True,
        )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = extract_choice_answer(resp)
        ok = pred == gold
        correct += ok
        total += 1

        cs = cat_stats.setdefault(cat, {"c": 0, "t": 0})
        cs["t"] += 1
        cs["c"] += ok

        results.append(dict(question=ex["question"], category=cat, gold=gold,
                            predicted=pred, correct=ok, response=resp))
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}] acc={correct}/{total}={correct/total:.4f}")

    elapsed = time.time() - t0
    acc = correct / total if total else 0
    print(f"\nMMLU-Pro [{tag}] {correct}/{total} = {acc:.4f}  ({elapsed:.0f}s)")
    for c in sorted(cat_stats):
        s = cat_stats[c]
        print(f"  {c}: {s['c']}/{s['t']}={s['c']/s['t']:.4f}" if s["t"] else "")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.output_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(dict(benchmark="mmlu_pro", tag=tag, mode=args.mode,
                       strategy=args.strategy, remask_threshold=args.remask_threshold,
                       accuracy=acc, correct=correct, total=total, time_s=elapsed,
                       category_stats=cat_stats), f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/mmlu_pro")
    p.add_argument("--gen_length", type=int, default=1024)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    run(p.parse_args())

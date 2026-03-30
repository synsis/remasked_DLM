"""
python -m eval.piqa --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.piqa --mode original

If the canonical `piqa` dataset fails (script-based on newer `datasets`),
the script falls back to `lmms-lab/piqa` split=test (same fields).
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
from remask.utils import extract_choice_answer, format_chat_prompt, tokenize_prompt
from eval.common import add_parallel_args, shard_dataset, _attach_gen_stats, aggregate_gen_stats


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


def _flush_summary(summary_path, tag, args, results, t0):
    correct = sum(1 for r in results if r.get("correct"))
    total = len(results)
    elapsed = time.time() - t0
    acc = correct / total if total else 0
    gen_agg = aggregate_gen_stats(results)
    with open(summary_path, "w") as f:
        summary = dict(benchmark="piqa", tag=tag, mode=args.mode,
                       strategy=args.strategy, remask_threshold=args.remask_threshold,
                       accuracy=acc, correct=correct, total=total, time_s=elapsed,
                       done=False)
        summary.update(gen_agg)
        json.dump(summary, f, indent=2)


def run(args):
    tag = run_tag(args)

    if args.mode == "original":
        model, tokenizer, mask_id = load_original_model(args.model_path)
    else:
        model, tokenizer, mask_id = load_remask_model(
            args.model_path, strategy=args.strategy,
            remask_threshold=args.remask_threshold)

    dataset = load_piqa()
    print(f"PIQA: {len(dataset)} items")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = shard_dataset(dataset, args)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_sfx = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.output_dir, f"{tag}{shard_sfx}_results.jsonl")
    summary_path = os.path.join(args.output_dir, f"{tag}{shard_sfx}_summary.json")
    correct = total = 0
    results = []
    t0 = time.time()

    fout = open(out_path, "w")
    for i, ex in enumerate(tqdm(dataset, desc=f"PIQA [{tag}]")):
        goal = ex["goal"]
        sol1, sol2 = ex["sol1"], ex["sol2"]
        gold = "A" if int(ex["label"]) == 0 else "B"
        user = (
            "Choose the better solution to achieve the goal.\n\n"
            f"Goal: {goal}\n\n"
            f"A. {sol1}\nB. {sol2}\n\n"
            "The answer is"
        )
        prompt = format_chat_prompt(user, tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(
            inputs=ids, gen_length=args.gen_length, block_length=args.block_length,
            steps=args.steps, threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            temperature=args.temperature, mask_id=mask_id, eos_early_stop=True,
        )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = extract_choice_answer(resp, 2)
        ok = pred == gold
        correct += ok
        total += 1
        r = dict(goal=goal, gold=gold, predicted=pred, correct=ok, response=resp)
        _attach_gen_stats(r, model)
        results.append(r)
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        fout.flush()
        if total % 10 == 0 or total == len(dataset):
            _flush_summary(summary_path, tag, args, results, t0)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] acc={correct}/{total}={correct/total:.4f}")
    fout.close()

    elapsed = time.time() - t0
    acc = correct / total if total else 0
    print(f"\nPIQA [{tag}] {correct}/{total} = {acc:.4f}  ({elapsed:.0f}s)")

    gen_agg = aggregate_gen_stats(results)
    with open(summary_path, "w") as f:
        summary = dict(benchmark="piqa", tag=tag, mode=args.mode,
                       strategy=args.strategy, remask_threshold=args.remask_threshold,
                       accuracy=acc, correct=correct, total=total, time_s=elapsed,
                       done=True)
        summary.update(gen_agg)
        json.dump(summary, f, indent=2)


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
    add_parallel_args(p)
    run(p.parse_args())

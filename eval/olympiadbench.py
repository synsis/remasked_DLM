"""
python -m eval.olympiadbench --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.olympiadbench --mode original
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
from remask.utils import (
    format_chat_prompt,
    tokenize_prompt,
    extract_boxed,
    extract_math_answer,
    normalize_math_answer,
)
from eval.common import add_parallel_args, shard_dataset

OLYMPIAD_PROMPT = (
    "Solve the following olympiad math problem step by step. "
    "Put your final answer in \\boxed{}.\n\n"
    "Problem: {problem}"
)


def run_tag(args):
    if args.mode == "original":
        return "original"
    return f"remask_{args.strategy}_{args.remask_threshold}"


def normalize_gold(ans):
    if ans is None:
        return ""
    if isinstance(ans, list):
        ans = ans[0] if ans else ""
    return str(ans).strip()


def load_olympiad_bench():
    last_err = None
    for path in (
        "AI-MO/olympiad-bench",
        "MathOdyssey/OlympiadBench",
        "math-ai/olympiadbench",
        "Hothan/OlympiadBench",
        "lmms-lab/OlympiadBench",
    ):
        try:
            return load_dataset(path, split="test"), path
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load OlympiadBench from Hub: {last_err}")


def problem_and_answer(ex):
    prob = ex.get("problem") or ex.get("question") or ""
    ans = ex.get("answer") or ex.get("final_answer")
    return prob, normalize_gold(ans)


def predict_answer(resp):
    boxed = extract_boxed(resp)
    raw = boxed if boxed is not None else extract_math_answer(resp)
    return raw


def run(args):
    tag = run_tag(args)

    if args.mode == "original":
        model, tokenizer, mask_id = load_original_model(args.model_path)
    else:
        model, tokenizer, mask_id = load_remask_model(
            args.model_path,
            strategy=args.strategy,
            remask_threshold=args.remask_threshold,
        )

    dataset, src = load_olympiad_bench()
    print(f"OlympiadBench: {len(dataset)} problems (source: {src})")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = shard_dataset(dataset, args)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_sfx = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.output_dir, f"{tag}{shard_sfx}_results.jsonl")
    correct = total = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(tqdm(dataset, desc=f"OlympiadBench [{tag}]")):
        prob, gold = problem_and_answer(ex)
        prompt = format_chat_prompt(OLYMPIAD_PROMPT.format(problem=prob), tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(
            inputs=ids,
            gen_length=args.gen_length,
            block_length=args.block_length,
            steps=args.steps,
            threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            temperature=args.temperature,
            mask_id=mask_id,
            eos_early_stop=True,
        )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = predict_answer(resp)
        ok = normalize_math_answer(pred) == normalize_math_answer(gold)
        correct += ok
        total += 1
        results.append(
            dict(problem=prob, gold=gold, predicted=pred, correct=ok, response=resp)
        )
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] acc={correct}/{total}={correct/total:.4f}")

    elapsed = time.time() - t0
    acc = correct / total if total else 0
    print(f"\nOlympiadBench [{tag}] {correct}/{total} = {acc:.4f}  ({elapsed:.0f}s)")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.output_dir, f"{tag}{shard_sfx}_summary.json"), "w") as f:
        json.dump(
            dict(
                benchmark="olympiadbench",
                tag=tag,
                mode=args.mode,
                strategy=args.strategy,
                remask_threshold=args.remask_threshold,
                hub_source=src,
                accuracy=acc,
                correct=correct,
                total=total,
                time_s=elapsed,
            ),
            f,
            indent=2,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument(
        "--strategy",
        choices=["low_prob", "t2t_remask", "logit_diff"],
        default="low_prob",
    )
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/olympiadbench")
    p.add_argument("--gen_length", type=int, default=1024)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    add_parallel_args(p)
    run(p.parse_args())

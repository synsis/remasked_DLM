"""
python -m eval.ifeval --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.ifeval --mode original
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
from remask.utils import format_chat_prompt, tokenize_prompt
from eval.common import add_parallel_args, shard_dataset


def load_ifeval_split():
    try:
        return load_dataset("google/IFEval", split="train")
    except Exception as e:
        print(f"  load_dataset('google/IFEval', split='train') failed: {e}")
        print("Install datasets and ensure IFEval is reachable on the hub.")
        raise


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

    dataset = load_ifeval_split()
    print(f"IFEval: {len(dataset)} prompts (train split; official scorer offline)")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = shard_dataset(dataset, args)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_sfx = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.output_dir, f"{tag}{shard_sfx}_results.jsonl")
    total = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(tqdm(dataset, desc=f"IFEval [{tag}]")):
        user = ex["prompt"]
        prompt = format_chat_prompt(user, tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(
            inputs=ids, gen_length=args.gen_length, block_length=args.block_length,
            steps=args.steps, threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            temperature=args.temperature, mask_id=mask_id, eos_early_stop=True,
        )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        total += 1
        results.append(dict(
            prompt=user,
            instruction_id_list=ex.get("instruction_id_list"),
            kwargs=ex.get("kwargs"),
            response=resp,
        ))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] saved={total}")

    elapsed = time.time() - t0
    print(f"\nIFEval [{tag}] saved {total} generations  ({elapsed:.0f}s)")
    print("Full IFEval scoring requires the official ifeval package / verifier.")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.output_dir, f"{tag}{shard_sfx}_summary.json"), "w") as f:
        json.dump(dict(
            benchmark="ifeval", tag=tag, mode=args.mode,
            strategy=args.strategy, remask_threshold=args.remask_threshold,
            main_metric="offline_ifeval_scorer",
            correct=None, total=total, time_s=elapsed,
            note="Run official IFEval scorer on results jsonl for metrics."),
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/ifeval")
    p.add_argument("--gen_length", type=int, default=1024)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    add_parallel_args(p)
    run(p.parse_args())

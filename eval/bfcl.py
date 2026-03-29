"""
python -m eval.bfcl --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.bfcl --mode original
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


def load_bfcl_split():
    last_err = None
    for name in ("gorilla-llm/Berkeley-Function-Calling-Leaderboard",):
        for split in ("test", "train"):
            try:
                return load_dataset(name, split=split), name, split
            except Exception as e:
                last_err = e
                print(f"  load_dataset({name!r}, split={split!r}) failed: {e}")
    print("Could not load BFCL dataset. Use BFCL / Gorilla eval tooling offline.")
    raise last_err


def user_message(ex):
    q = ex.get("question") or ex.get("user_message") or ex.get("prompt") or ""
    if q:
        return str(q)
    return json.dumps({k: ex[k] for k in sorted(ex.keys())}, ensure_ascii=False)[:12000]


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

    dataset, src_name, split_name = load_bfcl_split()
    print(f"BFCL ({src_name} {split_name}): {len(dataset)} rows (stub; score offline)")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = shard_dataset(dataset, args)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_sfx = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.output_dir, f"{tag}{shard_sfx}_results.jsonl")
    total = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(tqdm(dataset, desc=f"BFCL [{tag}]")):
        user = user_message(ex)
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
        row = dict(response=resp, raw_keys=list(ex.keys()))
        for k in ("question", "id", "function", "functions", "ground_truth", "answer"):
            if k in ex:
                v = ex[k]
                if isinstance(v, (dict, list)):
                    row[k] = v
                else:
                    row[k] = str(v)[:8000] if v is not None else None
        results.append(row)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] saved={total}")

    elapsed = time.time() - t0
    print(f"\nBFCL [{tag}] saved {total} generations  ({elapsed:.0f}s)")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.output_dir, f"{tag}{shard_sfx}_summary.json"), "w") as f:
        json.dump(dict(
            benchmark="bfcl", source=src_name, split=split_name, tag=tag,
            mode=args.mode, strategy=args.strategy,
            remask_threshold=args.remask_threshold,
            main_metric="offline_bfcl_scorer",
            correct=None, total=total, time_s=elapsed,
            note="Use Berkeley Function Calling Leaderboard official eval for metrics."),
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/bfcl")
    p.add_argument("--gen_length", type=int, default=512)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    add_parallel_args(p)
    run(p.parse_args())

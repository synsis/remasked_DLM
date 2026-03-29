"""
python -m eval.zebralogic --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.zebralogic --mode original
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
from remask.utils import format_chat_prompt, tokenize_prompt, extract_short_answer
from eval.common import add_parallel_args, shard_dataset


PROMPT_TPL = (
    "Solve the following logic puzzle step by step.\n\n{puzzle}\n\nAnswer:"
)


def load_zebra_split():
    last_err = None
    attempts = [
        ("yuchenlin/ZebraLogic", None, "test"),
        ("yuchenlin/ZebraLogic", None, "validation"),
        ("allenai/ZebraLogic", None, "test"),
        ("allenai/ZebraLogic", None, "validation"),
    ]
    for name, config, split in attempts:
        try:
            if config is None:
                return load_dataset(name, split=split), split
            return load_dataset(name, config, split=split), split
        except Exception as e:
            last_err = e
            print(f"  load_dataset({name!r}, split={split!r}) failed: {e}")
    print("Could not load ZebraLogic from known hubs.")
    raise last_err


def puzzle_text(ex):
    for k in ("puzzle", "question", "problem", "input"):
        v = ex.get(k)
        if v:
            return str(v)
    return json.dumps({k: ex[k] for k in ex.keys()}, ensure_ascii=False)[:8000]


def gold_answer(ex):
    for k in ("answer", "solution", "target", "output"):
        v = ex.get(k)
        if v is not None:
            return str(v).strip()
    return ""


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

    dataset, split_name = load_zebra_split()
    print(f"ZebraLogic ({split_name}): {len(dataset)} examples")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = shard_dataset(dataset, args)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_sfx = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.output_dir, f"{tag}{shard_sfx}_results.jsonl")
    correct = total = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(tqdm(dataset, desc=f"ZebraLogic [{tag}]")):
        puzzle = puzzle_text(ex)
        gold = gold_answer(ex)
        prompt = format_chat_prompt(PROMPT_TPL.format(puzzle=puzzle), tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(
            inputs=ids, gen_length=args.gen_length, block_length=args.block_length,
            steps=args.steps, threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            temperature=args.temperature, mask_id=mask_id, eos_early_stop=True,
        )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = extract_short_answer(resp)
        ok = bool(gold) and (gold.lower() in pred.lower() or pred.strip() == gold.strip())
        correct += int(ok)
        total += 1
        results.append(dict(puzzle=puzzle[:2000], gold=gold, predicted=pred,
                            correct=bool(ok), response=resp))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] acc={correct}/{total}={correct/total:.4f}")

    elapsed = time.time() - t0
    acc = correct / total if total else 0.0
    print(f"\nZebraLogic [{tag}] {correct}/{total} = {acc:.4f}  ({elapsed:.0f}s)")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.output_dir, f"{tag}{shard_sfx}_summary.json"), "w") as f:
        json.dump(dict(
            benchmark="zebralogic", split=split_name, tag=tag, mode=args.mode,
            strategy=args.strategy, remask_threshold=args.remask_threshold,
            main_metric="accuracy", accuracy=acc, correct=correct, total=total,
            time_s=elapsed), f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/zebralogic")
    p.add_argument("--gen_length", type=int, default=1024)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    add_parallel_args(p)
    run(p.parse_args())

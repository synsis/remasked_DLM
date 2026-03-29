"""
python -m eval.bbh_zh --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.bbh_zh --mode original
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
from eval.common import add_parallel_args, shard_dataset, _attach_gen_stats, aggregate_gen_stats


PROMPT_TPL = "请按照示例回答以下问题。\n\n{input}\n答案："


def load_bbh_zh_split():
    last_err = None
    specs = [
        ("opencompass/bbh_zh", None, "test"),
        ("opencompass/bbh_zh", None, "validation"),
        ("OpenCompass/bbh_zh", None, "test"),
    ]
    for name, config, split in specs:
        try:
            if config is None:
                ds = load_dataset(name, split=split)
            else:
                ds = load_dataset(name, config, split=split)
            return ds, name, split
        except Exception as e:
            last_err = e
            print(f"  load_dataset({name!r}, split={split!r}) failed: {e}")
    print("Could not load bbh_zh. Check OpenCompass mirror or dataset id.")
    raise last_err


def example_input(ex):
    for k in ("input", "question", "query", "text"):
        v = ex.get(k)
        if v:
            return str(v)
    return ""


def gold_target(ex):
    for k in ("target", "answer", "output", "label"):
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

    dataset, src_name, split_name = load_bbh_zh_split()
    print(f"BBH-Zh ({src_name} {split_name}): {len(dataset)} examples")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = shard_dataset(dataset, args)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_sfx = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.output_dir, f"{tag}{shard_sfx}_results.jsonl")
    correct = total = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(tqdm(dataset, desc=f"bbh_zh [{tag}]")):
        inp = example_input(ex)
        gold = gold_target(ex)
        prompt = format_chat_prompt(PROMPT_TPL.format(input=inp), tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(
            inputs=ids, gen_length=args.gen_length, block_length=args.block_length,
            steps=args.steps, threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            temperature=args.temperature, mask_id=mask_id, eos_early_stop=True,
        )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = extract_short_answer(resp)
        ok = bool(gold) and (gold in pred or pred.strip() == gold.strip())
        correct += int(ok)
        total += 1
        r = dict(input=inp[:4000], gold=gold, predicted=pred,
                 correct=bool(ok), response=resp)
        _attach_gen_stats(r, model)
        results.append(r)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] acc={correct}/{total}={correct/total:.4f}")

    elapsed = time.time() - t0
    acc = correct / total if total else 0.0
    print(f"\nBBH-Zh [{tag}] {correct}/{total} = {acc:.4f}  ({elapsed:.0f}s)")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    gen_agg = aggregate_gen_stats(results)
    with open(os.path.join(args.output_dir, f"{tag}{shard_sfx}_summary.json"), "w") as f:
        summary = dict(
            benchmark="bbh_zh", source=src_name, split=split_name, tag=tag,
            mode=args.mode, strategy=args.strategy,
            remask_threshold=args.remask_threshold,
            main_metric="accuracy", accuracy=acc, correct=correct, total=total,
            time_s=elapsed)
        summary.update(gen_agg)
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/bbh_zh")
    p.add_argument("--gen_length", type=int, default=16384)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    add_parallel_args(p)
    run(p.parse_args())

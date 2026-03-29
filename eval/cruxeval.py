"""
python -m eval.cruxeval --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.cruxeval --mode original
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
from eval.common import add_parallel_args, shard_dataset, _attach_gen_stats, aggregate_gen_stats


PROMPT_TPL = (
    "What is the output of the following Python code?\n\n"
    "```python\n{code}\n```\n\nInput: {input}\n\nOutput:"
)


def load_cruxeval_split():
    last_err = None
    specs = [
        ("cruxeval/cruxeval", None, "test"),
        ("flydust/CRUXEval-O", None, "test"),
        ("flydust/CRUXEval-O", None, "validation"),
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
    print("Could not load CRUXEval-O. Try another dataset revision.")
    raise last_err


def row_fields(ex):
    code = ex.get("code") or ex.get("program") or ""
    inp = ex.get("input")
    if inp is None:
        inp = ex.get("stdin") or ex.get("test_input") or ""
    out = ex.get("output")
    if out is None:
        out = ex.get("target") or ex.get("expected") or ""
    return str(code), str(inp), str(out).strip()


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

    dataset, src_name, split_name = load_cruxeval_split()
    print(f"CRUXEval-O ({src_name} {split_name}): {len(dataset)} examples")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = shard_dataset(dataset, args)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_sfx = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.output_dir, f"{tag}{shard_sfx}_results.jsonl")
    correct = total = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(tqdm(dataset, desc=f"CRUXEval [{tag}]")):
        code, inp, gold = row_fields(ex)
        prompt = format_chat_prompt(
            PROMPT_TPL.format(code=code, input=inp), tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(
            inputs=ids, gen_length=args.gen_length, block_length=args.block_length,
            steps=args.steps, threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            temperature=args.temperature, mask_id=mask_id, eos_early_stop=True,
        )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        pred_full = resp.strip()
        pred_line = pred_full.splitlines()[-1].strip() if pred_full else ""
        g = gold.strip()
        ok = bool(g) and (pred_full == g or pred_line == g)
        correct += int(ok)
        total += 1
        r = dict(
            code=code[:4000], input=inp, gold=gold, predicted=pred_full,
            correct=bool(ok), response=resp)
        _attach_gen_stats(r, model)
        results.append(r)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] acc={correct}/{total}={correct/total:.4f}")

    elapsed = time.time() - t0
    acc = correct / total if total else 0.0
    print(f"\nCRUXEval-O [{tag}] {correct}/{total} = {acc:.4f}  ({elapsed:.0f}s)")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    gen_agg = aggregate_gen_stats(results)
    with open(os.path.join(args.output_dir, f"{tag}{shard_sfx}_summary.json"), "w") as f:
        summary = dict(
            benchmark="cruxeval", source=src_name, split=split_name, tag=tag,
            mode=args.mode, strategy=args.strategy,
            remask_threshold=args.remask_threshold,
            main_metric="exact_match", accuracy=acc, correct=correct, total=total,
            time_s=elapsed)
        summary.update(gen_agg)
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/cruxeval")
    p.add_argument("--gen_length", type=int, default=256)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    add_parallel_args(p)
    run(p.parse_args())

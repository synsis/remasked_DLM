"""
python -m eval.bbh_extra_hard --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.bbh_extra_hard --mode original

Fallback local JSONL: one object per line with keys input, target (same as BBH).
Place at eval/bbeh_local.jsonl if Hub datasets are unavailable.
"""

import argparse
import json
import os
import remask.env  # noqa: F401
import re
import string
import time
import unicodedata

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from remask import load_remask_model, load_original_model
from remask.utils import compute_em, format_chat_prompt, tokenize_prompt
from eval.common import add_parallel_args, shard_dataset


def _norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def bbh_style_correct(resp, target):
    if compute_em(resp, target) >= 1.0:
        return True
    rn, tn = _norm(resp), _norm(target)
    if not tn:
        return False
    return tn in rn or rn == tn


def load_bbeh():
    for name, split in [
        ("BBEH/bbeh", "train"),
        ("kazemi/bbeh", "test"),
        ("google/bbeh", "test"),
    ]:
        try:
            return load_dataset(name, split=split)
        except Exception:
            continue
    local = os.path.join(os.path.dirname(__file__), "bbeh_local.jsonl")
    if not os.path.isfile(local):
        raise FileNotFoundError(
            "BBEH: no Hub dataset worked; add eval/bbeh_local.jsonl "
            "(JSONL lines with input, target) or install a compatible dataset."
        )
    rows = []
    with open(local, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


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

    dataset = load_bbeh()
    print(f"BBEH: {len(dataset)} items")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = shard_dataset(dataset, args)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_sfx = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.output_dir, f"{tag}{shard_sfx}_results.jsonl")
    correct = total = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(tqdm(dataset, desc=f"BBEH [{tag}]")):
        inp = ex["input"]
        target = ex["target"]
        user = f"Follow the given examples and answer the question.\n\n{inp}\nAnswer:"
        prompt = format_chat_prompt(user, tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(
            inputs=ids, gen_length=args.gen_length, block_length=args.block_length,
            steps=args.steps, threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            temperature=args.temperature, mask_id=mask_id, eos_early_stop=True,
        )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        ok = bbh_style_correct(resp, target)
        correct += ok
        total += 1
        results.append(dict(input=inp, target=target, response=resp, correct=ok))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] acc={correct}/{total}={correct/total:.4f}")

    elapsed = time.time() - t0
    acc = correct / total if total else 0
    print(f"\nBBEH [{tag}] {correct}/{total} = {acc:.4f}  ({elapsed:.0f}s)")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.output_dir, f"{tag}{shard_sfx}_summary.json"), "w") as f:
        json.dump(dict(benchmark="bbh_extra_hard", tag=tag, mode=args.mode,
                       strategy=args.strategy, remask_threshold=args.remask_threshold,
                       accuracy=acc, correct=correct, total=total, time_s=elapsed), f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/bbh_extra_hard")
    p.add_argument("--gen_length", type=int, default=512)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    add_parallel_args(p)
    run(p.parse_args())

"""
python -m eval.kor_bench --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.kor_bench --mode original

If Hub datasets are unavailable, add eval/kor_bench_local.jsonl with one JSON object
per line: {"question": "...", "answer": "..."} (answer is the reference string).
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


def kor_correct(resp, answer):
    if compute_em(resp, answer) >= 1.0:
        return True
    rn, an = _norm(resp), _norm(answer)
    if not an:
        return False
    return an in rn or rn == an


def load_kor_bench():
    for name, split in [
        ("kaist-ai/KOR-Bench", "test"),
        ("KOR-Bench/KOR-Bench", "test"),
    ]:
        try:
            return load_dataset(name, split=split)
        except Exception:
            continue
    local = os.path.join(os.path.dirname(__file__), "kor_bench_local.jsonl")
    if not os.path.isfile(local):
        raise FileNotFoundError(
            "KOR-Bench: Hub load failed; place eval/kor_bench_local.jsonl "
            "with JSONL lines {\"question\":..., \"answer\":...}."
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

    dataset = load_kor_bench()
    print(f"KOR-Bench: {len(dataset)} items")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = shard_dataset(dataset, args)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_sfx = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.output_dir, f"{tag}{shard_sfx}_results.jsonl")
    correct = total = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(tqdm(dataset, desc=f"KOR-Bench [{tag}]")):
        q = ex["question"]
        ans = ex["answer"]
        user = f"{q}\n\nAnswer:"
        prompt = format_chat_prompt(user, tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(
            inputs=ids, gen_length=args.gen_length, block_length=args.block_length,
            steps=args.steps, threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            temperature=args.temperature, mask_id=mask_id, eos_early_stop=True,
        )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        ok = kor_correct(resp, ans)
        correct += ok
        total += 1
        results.append(dict(question=q, answer=ans, response=resp, correct=ok))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] acc={correct}/{total}={correct/total:.4f}")

    elapsed = time.time() - t0
    acc = correct / total if total else 0
    print(f"\nKOR-Bench [{tag}] {correct}/{total} = {acc:.4f}  ({elapsed:.0f}s)")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.output_dir, f"{tag}{shard_sfx}_summary.json"), "w") as f:
        json.dump(dict(benchmark="kor_bench", tag=tag, mode=args.mode,
                       strategy=args.strategy, remask_threshold=args.remask_threshold,
                       accuracy=acc, correct=correct, total=total, time_s=elapsed), f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/kor_bench")
    p.add_argument("--gen_length", type=int, default=512)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    add_parallel_args(p)
    run(p.parse_args())

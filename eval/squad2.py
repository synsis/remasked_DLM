"""
python -m eval.squad2 --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.squad2 --mode original
"""

import argparse
import json
import os
import remask.env  # noqa: F401
import re
import time

import torch
from datasets import load_dataset
from tqdm import tqdm

from remask import load_remask_model, load_original_model
from remask.utils import (
    format_chat_prompt,
    tokenize_prompt,
    extract_short_answer,
    compute_f1,
    max_metric_over_answers,
)
from eval.common import add_parallel_args, shard_dataset, _attach_gen_stats, aggregate_gen_stats

PROMPT_TPL = (
    "Read the context and answer the question. If unanswerable, say 'unanswerable'.\n\n"
    "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
)

UNANSWERABLE_PAT = re.compile(
    r"\b(unanswerable|not answerable|no answer|cannot answer|can't answer|"
    r"impossible to answer|无法回答|没有答案|不可回答)\b",
    re.IGNORECASE,
)


def load_squad2_split():
    last_err = None
    for name in ("rajpurkar/squad_v2", "squad_v2"):
        try:
            return load_dataset(name, split="validation")
        except Exception as e:
            last_err = e
            print(f"  load_dataset({name!r}, split='validation') failed: {e}")
    print("Could not load SQuAD 2.0. Check dataset id and connectivity.")
    raise last_err


def is_unanswerable_prediction(text):
    t = (text or "").strip().lower()
    if not t:
        return True
    return UNANSWERABLE_PAT.search(t) is not None


def squad2_f1_score(pred_raw, gold_texts):
    pred = extract_short_answer(pred_raw)
    if not gold_texts:
        return 1.0 if is_unanswerable_prediction(pred_raw) else 0.0
    return max_metric_over_answers(pred, gold_texts, compute_f1)


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

    dataset = load_squad2_split()
    print(f"SQuAD 2.0: {len(dataset)} examples")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = shard_dataset(dataset, args)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_sfx = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.output_dir, f"{tag}{shard_sfx}_results.jsonl")
    sum_f1 = 0.0
    total = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(tqdm(dataset, desc=f"SQuAD2 [{tag}]")):
        context = ex["context"]
        question = ex["question"]
        gold_texts = list((ex.get("answers") or {}).get("text") or [])
        prompt = format_chat_prompt(
            PROMPT_TPL.format(context=context, question=question), tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(
            inputs=ids, gen_length=args.gen_length, block_length=args.block_length,
            steps=args.steps, threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            temperature=args.temperature, mask_id=mask_id, eos_early_stop=True,
        )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        f1 = squad2_f1_score(resp, gold_texts)
        sum_f1 += f1
        total += 1
        r = dict(
            context=context, question=question, gold=gold_texts,
            predicted=extract_short_answer(resp), f1=f1, response=resp)
        _attach_gen_stats(r, model)
        results.append(r)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] F1={sum_f1/total:.4f}")

    elapsed = time.time() - t0
    avg_f1 = sum_f1 / total if total else 0.0
    perfect_f1 = sum(1 for r in results if r["f1"] >= 1.0 - 1e-9)
    print(f"\nSQuAD 2.0 [{tag}] F1={avg_f1:.4f}  ({elapsed:.0f}s)")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    gen_agg = aggregate_gen_stats(results)
    with open(os.path.join(args.output_dir, f"{tag}{shard_sfx}_summary.json"), "w") as f:
        summary = dict(
            benchmark="squad2", tag=tag, mode=args.mode,
            strategy=args.strategy, remask_threshold=args.remask_threshold,
            main_metric="f1", f1=avg_f1,
            correct=perfect_f1, total=total, time_s=elapsed)
        summary.update(gen_agg)
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/squad2")
    p.add_argument("--gen_length", type=int, default=16384)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    add_parallel_args(p)
    run(p.parse_args())

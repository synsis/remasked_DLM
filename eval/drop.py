"""
python -m eval.drop --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.drop --mode original
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
    extract_short_answer,
    compute_f1,
    compute_em,
    max_metric_over_answers,
)

PROMPT_TPL = (
    "Read the passage and answer the question.\n\n"
    "Passage: {passage}\n\nQuestion: {question}\n\nAnswer:"
)


def load_drop_split():
    last_err = None
    for spec in (
        ("drop", None, "validation"),
        ("ucinlp/drop", None, "validation"),
    ):
        name, config, split = spec
        try:
            if config is None:
                return load_dataset(name, split=split)
            return load_dataset(name, config, split=split)
        except Exception as e:
            last_err = e
            print(f"  load_dataset({name!r}, split={split!r}) failed: {e}")
    print("Could not load DROP. Try: pip install datasets && check HF mirror connectivity.")
    raise last_err


def gold_spans(ex):
    asp = ex.get("answers_spans") or {}
    spans = asp.get("spans")
    if spans is None:
        return []
    return list(spans)


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

    dataset = load_drop_split()
    print(f"DROP: {len(dataset)} examples")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{tag}_results.jsonl")
    sum_em = sum_f1 = 0.0
    total = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(tqdm(dataset, desc=f"DROP [{tag}]")):
        passage = ex["passage"]
        question = ex["question"]
        gold = gold_spans(ex)
        prompt = format_chat_prompt(
            PROMPT_TPL.format(passage=passage, question=question), tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(
            inputs=ids, gen_length=args.gen_length, block_length=args.block_length,
            steps=args.steps, threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            temperature=args.temperature, mask_id=mask_id, eos_early_stop=True,
        )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = extract_short_answer(resp)
        em = max_metric_over_answers(pred, gold, compute_em) if gold else 0.0
        f1 = max_metric_over_answers(pred, gold, compute_f1) if gold else 0.0
        sum_em += em
        sum_f1 += f1
        total += 1
        results.append(dict(
            passage=passage, question=question, gold=gold, predicted=pred,
            em=em, f1=f1, response=resp))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] EM={sum_em/total:.4f} F1={sum_f1/total:.4f}")

    elapsed = time.time() - t0
    avg_em = sum_em / total if total else 0.0
    avg_f1 = sum_f1 / total if total else 0.0
    em_correct = sum(1 for r in results if r["em"] >= 1.0 - 1e-9)
    print(f"\nDROP [{tag}] EM={avg_em:.4f} F1={avg_f1:.4f}  ({elapsed:.0f}s)")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.output_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(dict(
            benchmark="drop", tag=tag, mode=args.mode,
            strategy=args.strategy, remask_threshold=args.remask_threshold,
            main_metric="f1", em=avg_em, f1=avg_f1,
            correct=em_correct, total=total, time_s=elapsed), f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/drop")
    p.add_argument("--gen_length", type=int, default=256)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    run(p.parse_args())

"""
python -m eval.gsm_plus --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.gsm_plus --mode original
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
from remask.utils import format_chat_prompt, tokenize_prompt, extract_math_answer, normalize_numeric

GSM_PLUS_PROMPT = (
    "Solve the following math problem step by step. "
    "Show your work, then put your final numeric answer after ####.\n\n"
    "Problem: {question}"
)


def extract_gold(answer_text):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", str(answer_text))
    if m:
        return m.group(1).replace(",", "").strip()
    return str(answer_text).replace(",", "").strip()


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
            args.model_path,
            strategy=args.strategy,
            remask_threshold=args.remask_threshold,
        )

    try:
        dataset = load_dataset("qintongli/GSM-Plus", split="testmini")
    except Exception:
        dataset = load_dataset("qintongli/GSM-Plus", split="test")
    print(f"GSM-Plus: {len(dataset)} problems")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{tag}_results.jsonl")
    correct = total = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(tqdm(dataset, desc=f"GSM-Plus [{tag}]")):
        gold = extract_gold(ex["answer"])
        prompt = format_chat_prompt(
            GSM_PLUS_PROMPT.format(question=ex["question"]), tokenizer
        )
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
        pred = extract_math_answer(resp)
        ok = normalize_numeric(pred) == normalize_numeric(gold)
        correct += ok
        total += 1
        results.append(
            dict(
                question=ex["question"],
                gold=gold,
                predicted=pred,
                correct=ok,
                response=resp,
            )
        )
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] acc={correct}/{total}={correct/total:.4f}")

    elapsed = time.time() - t0
    acc = correct / total if total else 0
    print(f"\nGSM-Plus [{tag}] {correct}/{total} = {acc:.4f}  ({elapsed:.0f}s)")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.output_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(
            dict(
                benchmark="gsm_plus",
                tag=tag,
                mode=args.mode,
                strategy=args.strategy,
                remask_threshold=args.remask_threshold,
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
    p.add_argument("--output_dir", default="results/gsm_plus")
    p.add_argument("--gen_length", type=int, default=512)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    run(p.parse_args())

"""
python -m eval.cmath --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.cmath --mode original
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
from remask.utils import format_chat_prompt, tokenize_prompt, extract_math_answer, normalize_numeric

CMATH_PROMPT = (
    "请逐步解决以下数学问题，在最后用####给出最终数字答案。\n\n问题：{question}"
)


def get_gold(ex):
    raw = ex.get("golden")
    if raw is None:
        raw = ex.get("answer")
    return str(raw).strip() if raw is not None else ""


def run_tag(args):
    if args.mode == "original":
        return "original"
    return f"remask_{args.strategy}_{args.remask_threshold}"


def load_cmath():
    last_err = None
    for path in ("weitianwen/cmath", "CMATH/CMATH"):
        try:
            return load_dataset(path, split="test")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load cmath dataset: {last_err}")


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

    dataset = load_cmath()
    print(f"cmath: {len(dataset)} problems")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{tag}_results.jsonl")
    correct = total = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(tqdm(dataset, desc=f"cmath [{tag}]")):
        gold = get_gold(ex)
        prompt = format_chat_prompt(
            CMATH_PROMPT.format(question=ex["question"]), tokenizer
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
    print(f"\ncmath [{tag}] {correct}/{total} = {acc:.4f}  ({elapsed:.0f}s)")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.output_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(
            dict(
                benchmark="cmath",
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
    p.add_argument("--output_dir", default="results/cmath")
    p.add_argument("--gen_length", type=int, default=512)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    run(p.parse_args())

"""
python -m eval.gsm_plus --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.gsm_plus --mode original
python -m eval.gsm_plus --mode original --batch_size 4       # batched inference
python -m eval.gsm_plus --mode remask --shard_id 0 --num_shards 2  # data parallel
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
    tag = f"remask_{args.strategy}_{args.remask_threshold}"
    if hasattr(args, 'max_remask_ratio') and args.max_remask_ratio != 0.25:
        tag += f"_r{args.max_remask_ratio}"
    return tag


def gen_kwargs(args, mask_id):
    return dict(
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        threshold=args.threshold,
        editing_threshold=args.editing_threshold,
        temperature=args.temperature,
        mask_id=mask_id,
        eos_early_stop=True,
    )


def run(args):
    tag = run_tag(args)

    if args.mode == "original":
        model, tokenizer, mask_id = load_original_model(args.model_path)
    else:
        model, tokenizer, mask_id = load_remask_model(
            args.model_path,
            strategy=args.strategy,
            remask_threshold=args.remask_threshold,
            max_remask_ratio=args.max_remask_ratio,
        )

    try:
        dataset = load_dataset("qintongli/GSM-Plus", split="testmini")
    except Exception:
        dataset = load_dataset("qintongli/GSM-Plus", split="test")
    print(f"GSM-Plus: {len(dataset)} problems")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    if args.num_shards > 1:
        indices = list(range(args.shard_id, len(dataset), args.num_shards))
        dataset = dataset.select(indices)
        print(f"  shard {args.shard_id}/{args.num_shards}: {len(dataset)} samples")

    os.makedirs(args.output_dir, exist_ok=True)
    shard_suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.output_dir, f"{tag}{shard_suffix}_results.jsonl")

    correct = total = 0
    results = []
    t0 = time.time()
    gkw = gen_kwargs(args, mask_id)
    batch_size = args.batch_size

    if batch_size > 1:
        _run_batched(model, tokenizer, dataset, batch_size, gkw, results, args, tag)
    else:
        _run_sequential(model, tokenizer, dataset, gkw, results, args, tag)

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    elapsed = time.time() - t0
    acc = correct / total if total else 0
    print(f"\nGSM-Plus [{tag}] {correct}/{total} = {acc:.4f}  ({elapsed:.0f}s)")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    summary_path = os.path.join(
        args.output_dir, f"{tag}{shard_suffix}_summary.json",
    )
    with open(summary_path, "w") as f:
        json.dump(
            dict(
                benchmark="gsm_plus",
                tag=tag,
                mode=args.mode,
                accuracy=acc,
                correct=correct,
                total=total,
                time_s=elapsed,
                gen_length=args.gen_length,
                block_length=args.block_length,
                steps=args.steps,
                threshold=args.threshold,
                editing_threshold=args.editing_threshold,
                temperature=args.temperature,
                strategy=args.strategy,
                remask_threshold=args.remask_threshold,
                max_remask_ratio=getattr(args, 'max_remask_ratio', None),
                max_remask_per_pos=getattr(args, 'max_remask_per_pos', None),
                batch_size=batch_size,
                shard_id=args.shard_id if args.num_shards > 1 else None,
                num_shards=args.num_shards if args.num_shards > 1 else None,
            ),
            f,
            indent=2,
        )


def _run_sequential(model, tokenizer, dataset, gkw, results, args, tag):
    correct = total = 0
    for i, ex in enumerate(tqdm(dataset, desc=f"GSM-Plus [{tag}]")):
        gold = extract_gold(ex["answer"])
        prompt = format_chat_prompt(
            GSM_PLUS_PROMPT.format(question=ex["question"]), tokenizer
        )
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(inputs=ids, **gkw)
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


def _run_batched(model, tokenizer, dataset, batch_size, gkw, results, args, tag):
    correct = total = 0
    examples = list(dataset)
    for start in tqdm(range(0, len(examples), batch_size),
                      desc=f"GSM-Plus [{tag}] bs={batch_size}"):
        batch_ex = examples[start : start + batch_size]
        golds, prompts_ids = [], []
        for ex in batch_ex:
            golds.append(extract_gold(ex["answer"]))
            prompt = format_chat_prompt(
                GSM_PLUS_PROMPT.format(question=ex["question"]), tokenizer
            )
            prompts_ids.append(tokenize_prompt(prompt, tokenizer, model.device))

        outputs = model.generate_batch(inputs_list=prompts_ids, **gkw)

        for j, (out, gold, ex) in enumerate(zip(outputs, golds, batch_ex)):
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

        if total > 0 and total % 50 < batch_size:
            print(f"  [{total}] acc={correct}/{total}={correct/total:.4f}")


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
    p.add_argument("--max_remask_ratio", type=float, default=0.25,
                    help="Max fraction of editable tokens to remask per step")
    p.add_argument("--output_dir", default="results/gsm_plus")
    p.add_argument("--gen_length", type=int, default=512)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    # Parallelism options
    p.add_argument("--batch_size", type=int, default=1,
                    help="Number of prompts to process simultaneously (batched forward pass)")
    p.add_argument("--shard_id", type=int, default=0,
                    help="Dataset shard index for multi-process parallelism")
    p.add_argument("--num_shards", type=int, default=1,
                    help="Total number of shards for multi-process parallelism")
    run(p.parse_args())

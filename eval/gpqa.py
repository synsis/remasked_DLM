import argparse
import json
import os
import remask.env  # noqa: F401
import time

from datasets import load_dataset
from tqdm import tqdm

from remask import load_remask_model, load_original_model
from remask.utils import format_chat_prompt, tokenize_prompt, extract_choice_answer
from eval.common import add_parallel_args, shard_dataset

GPQA_PROMPT = (
    'The following is a multiple choice question from a graduate-level exam. '
    'Think step by step and then output the answer in the format "The answer is (X)" '
    "where X is the correct letter.\n\nQuestion: {Question}"
)


def run_tag(args):
    if args.mode == "original":
        return "original"
    return f"remask_{args.strategy}_{args.remask_threshold}"


def norm_gold_letter(s):
    s = str(s).strip().upper()
    for ch in s:
        if ch in "ABCD":
            return ch
    return s[:1] if s else ""


def run(args):
    tag = run_tag(args)

    if args.mode == "original":
        model, tokenizer, mask_id = load_original_model(args.model_path)
    else:
        model, tokenizer, mask_id = load_remask_model(
            args.model_path, strategy=args.strategy,
            remask_threshold=args.remask_threshold)

    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    print(f"GPQA diamond: {len(dataset)} items")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = shard_dataset(dataset, args)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_sfx = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.output_dir, f"{tag}{shard_sfx}_results.jsonl")
    correct = total = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(tqdm(dataset, desc=f"GPQA [{tag}]")):
        gold = norm_gold_letter(ex["Correct Answer"])
        user = GPQA_PROMPT.format(Question=ex["Question"])
        prompt = format_chat_prompt(user, tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(
            inputs=ids, gen_length=args.gen_length, block_length=args.block_length,
            steps=args.steps, threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            temperature=args.temperature, mask_id=mask_id, eos_early_stop=True,
        )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = extract_choice_answer(resp, 4)
        ok = pred == gold
        correct += ok
        total += 1
        results.append(dict(
            gold=gold, predicted=pred, correct=ok, response=resp,
            question=ex["Question"]))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] acc={correct}/{total}={correct/total:.4f}")

    elapsed = time.time() - t0
    acc = correct / total if total else 0
    print(f"\nGPQA [{tag}] {correct}/{total} = {acc:.4f}  ({elapsed:.0f}s)")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.output_dir, f"{tag}{shard_sfx}_summary.json"), "w") as f:
        json.dump(dict(
            benchmark="gpqa", tag=tag, mode=args.mode,
            strategy=args.strategy, remask_threshold=args.remask_threshold,
            accuracy=acc, correct=correct, total=total, time_s=elapsed), f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/gpqa")
    p.add_argument("--gen_length", type=int, default=1024)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    add_parallel_args(p)
    run(p.parse_args())

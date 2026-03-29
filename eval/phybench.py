import argparse
import json
import os
import remask.env  # noqa: F401
import time

from datasets import Dataset, load_dataset
from tqdm import tqdm

from remask import load_remask_model, load_original_model
from remask.utils import (
    format_chat_prompt,
    tokenize_prompt,
    extract_boxed,
    normalize_math_answer,
)
from eval.common import add_parallel_args, shard_dataset

PHY_PROMPT = (
    "Solve the following physics problem step by step. "
    "Put your final answer in \\boxed{}.\n\nProblem: {problem}"
)

_LOCAL_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "phybench.json")


def run_tag(args):
    if args.mode == "original":
        return "original"
    return f"remask_{args.strategy}_{args.remask_threshold}"


def load_phybench():
    try:
        return load_dataset("Phy-Bench/PHYBench", split="test")
    except Exception:
        with open(_LOCAL_JSON, "r") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            rows = raw
        elif isinstance(raw, dict):
            rows = None
            for k in ("data", "test", "examples"):
                v = raw.get(k)
                if isinstance(v, list):
                    rows = v
                    break
            if rows is None:
                raise RuntimeError(
                    f"{_LOCAL_JSON}: need a JSON list or dict with data/test/examples list")
        else:
            raise RuntimeError(f"{_LOCAL_JSON}: unsupported JSON root {type(raw)}")
        return Dataset.from_list(rows)


def run(args):
    tag = run_tag(args)

    if args.mode == "original":
        model, tokenizer, mask_id = load_original_model(args.model_path)
    else:
        model, tokenizer, mask_id = load_remask_model(
            args.model_path, strategy=args.strategy,
            remask_threshold=args.remask_threshold)

    dataset = load_phybench()
    print(f"PHYBench: {len(dataset)} items")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = shard_dataset(dataset, args)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_sfx = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.output_dir, f"{tag}{shard_sfx}_results.jsonl")
    correct = total = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(tqdm(dataset, desc=f"PHYBench [{tag}]")):
        problem = ex.get("problem") or ex.get("question") or ex.get("Problem") or ""
        gold_raw = ex.get("answer") or ex.get("Answer") or ""
        user = PHY_PROMPT.format(problem=problem)
        prompt = format_chat_prompt(user, tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(
            inputs=ids, gen_length=args.gen_length, block_length=args.block_length,
            steps=args.steps, threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            temperature=args.temperature, mask_id=mask_id, eos_early_stop=True,
        )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        boxed = extract_boxed(resp)
        pred_norm = normalize_math_answer(boxed if boxed is not None else "")
        gold_norm = normalize_math_answer(str(gold_raw))
        ok = pred_norm == gold_norm
        correct += ok
        total += 1
        results.append(dict(
            gold=str(gold_raw), predicted_boxed=boxed, correct=ok, response=resp,
            problem=problem))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] acc={correct}/{total}={correct/total:.4f}")

    elapsed = time.time() - t0
    acc = correct / total if total else 0
    print(f"\nPHYBench [{tag}] {correct}/{total} = {acc:.4f}  ({elapsed:.0f}s)")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.output_dir, f"{tag}{shard_sfx}_summary.json"), "w") as f:
        json.dump(dict(
            benchmark="phybench", tag=tag, mode=args.mode,
            strategy=args.strategy, remask_threshold=args.remask_threshold,
            accuracy=acc, correct=correct, total=total, time_s=elapsed), f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/phybench")
    p.add_argument("--gen_length", type=int, default=1024)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    add_parallel_args(p)
    run(p.parse_args())

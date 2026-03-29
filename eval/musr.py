"""
python -m eval.musr --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.musr --mode original
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
from remask.utils import format_chat_prompt, tokenize_prompt, extract_choice_answer
from eval.common import add_parallel_args, shard_dataset, _attach_gen_stats, aggregate_gen_stats


def load_musr_split():
    last_err = None
    for name in ("sprague/MuSR", "TAUR-Lab/MuSR"):
        try:
            return load_dataset(name, split="validation")
        except Exception as e:
            last_err = e
            print(f"  load_dataset({name!r}, split='validation') failed: {e}")
    print("Could not load MuSR. Try another revision or local path.")
    raise last_err


def build_prompt(ex):
    q = ex.get("question") or ex.get("query") or ""
    choices = ex.get("choices") or ex.get("options")
    if choices and isinstance(choices, (list, tuple)) and len(choices) > 0:
        letters = "ABCDEFGHIJ"
        lines = []
        for j, c in enumerate(choices):
            if j < len(letters):
                lines.append(f"{letters[j]}. {c}")
        choice_block = "\n".join(lines)
        return (
            f"{q}\n\n{choice_block}\n\n"
            "Think step by step, then answer with the letter in the form "
            '"The answer is (X)".'
        )
    return q


def normalize_gold_answer(ex):
    ans = ex.get("answer")
    if ans is None:
        return ""
    if isinstance(ans, int):
        return "ABCDEFGHIJ"[ans] if 0 <= ans < 10 else str(ans)
    s = str(ans).strip()
    if len(s) == 1 and s.upper() in "ABCDEFGHIJ":
        return s.upper()
    return s


def score_musr(pred_text, gold, has_choices, n_choices):
    if has_choices:
        pred = extract_choice_answer(pred_text, n_choices=n_choices)
        return 1.0 if pred and gold and pred == gold.upper() else 0.0
    pred = pred_text.strip()
    g = str(gold).strip()
    if not g:
        return 0.0
    return 1.0 if g.lower() in pred.lower() or pred.strip() == g else 0.0


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

    dataset = load_musr_split()
    print(f"MuSR: {len(dataset)} examples")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = shard_dataset(dataset, args)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_sfx = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.output_dir, f"{tag}{shard_sfx}_results.jsonl")
    correct = total = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(tqdm(dataset, desc=f"MuSR [{tag}]")):
        user = build_prompt(ex)
        choices = ex.get("choices") or ex.get("options")
        n_ch = len(choices) if isinstance(choices, (list, tuple)) else 10
        n_ch = max(2, min(n_ch, 10))
        gold = normalize_gold_answer(ex)
        has_mc = bool(choices and isinstance(choices, (list, tuple)) and len(choices) > 0)

        prompt = format_chat_prompt(user, tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(
            inputs=ids, gen_length=args.gen_length, block_length=args.block_length,
            steps=args.steps, threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            temperature=args.temperature, mask_id=mask_id, eos_early_stop=True,
        )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        ok = score_musr(resp, gold, has_mc, n_ch)
        correct += int(ok)
        total += 1
        r = dict(
            gold=gold, predicted=extract_choice_answer(resp, n_choices=n_ch) if has_mc else resp[:500],
            correct=bool(ok), response=resp)
        _attach_gen_stats(r, model)
        results.append(r)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] acc={correct}/{total}={correct/total:.4f}")

    elapsed = time.time() - t0
    acc = correct / total if total else 0.0
    print(f"\nMuSR [{tag}] {correct}/{total} = {acc:.4f}  ({elapsed:.0f}s)")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    gen_agg = aggregate_gen_stats(results)
    with open(os.path.join(args.output_dir, f"{tag}{shard_sfx}_summary.json"), "w") as f:
        summary = dict(
            benchmark="musr", tag=tag, mode=args.mode,
            strategy=args.strategy, remask_threshold=args.remask_threshold,
            main_metric="accuracy", accuracy=acc, correct=correct, total=total,
            time_s=elapsed)
        summary.update(gen_agg)
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/musr")
    p.add_argument("--gen_length", type=int, default=512)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    add_parallel_args(p)
    run(p.parse_args())

import argparse
import json
import os
import remask.env  # noqa: F401
import time

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

TRIVIA_PROMPT = "Answer the following trivia question concisely.\n\nQuestion: {question}\nAnswer:"


def run_tag(args):
    if args.mode == "original":
        return "original"
    return f"remask_{args.strategy}_{args.remask_threshold}"


def gold_aliases(ex):
    ans = ex["answer"]
    aliases = list(ans.get("aliases") or [])
    val = ans.get("value")
    ordered = aliases + ([val] if val is not None else [])
    seen = set()
    out = []
    for x in ordered:
        if x is None:
            continue
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def run(args):
    tag = run_tag(args)

    if args.mode == "original":
        model, tokenizer, mask_id = load_original_model(args.model_path)
    else:
        model, tokenizer, mask_id = load_remask_model(
            args.model_path, strategy=args.strategy,
            remask_threshold=args.remask_threshold)

    dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    print(f"TriviaQA rc.nocontext validation: {len(dataset)} items")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{tag}_results.jsonl")
    sum_em = sum_f1 = 0.0
    total = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(tqdm(dataset, desc=f"TriviaQA [{tag}]")):
        golds = gold_aliases(ex)
        user = TRIVIA_PROMPT.format(question=ex["question"])
        prompt = format_chat_prompt(user, tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(
            inputs=ids, gen_length=args.gen_length, block_length=args.block_length,
            steps=args.steps, threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            temperature=args.temperature, mask_id=mask_id, eos_early_stop=True,
        )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = extract_short_answer(resp)
        em = max_metric_over_answers(pred, golds, compute_em)
        f1 = max_metric_over_answers(pred, golds, compute_f1)
        sum_em += em
        sum_f1 += f1
        total += 1
        results.append(dict(
            golds=golds, predicted=pred, em=em, f1=f1, response=resp,
            question=ex["question"]))
        if (i + 1) % 50 == 0:
            print(
                f"  [{i+1}] EM={sum_em/total:.4f} F1={sum_f1/total:.4f} "
                f"({total} samples)")

    elapsed = time.time() - t0
    mean_em = sum_em / total if total else 0.0
    mean_f1 = sum_f1 / total if total else 0.0
    print(f"\nTriviaQA [{tag}] EM={mean_em:.4f} F1={mean_f1:.4f}  ({elapsed:.0f}s)")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.output_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(dict(
            benchmark="triviaqa", tag=tag, mode=args.mode,
            strategy=args.strategy, remask_threshold=args.remask_threshold,
            em=mean_em, f1=mean_f1, total=total, time_s=elapsed), f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/triviaqa")
    p.add_argument("--gen_length", type=int, default=256)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    run(p.parse_args())

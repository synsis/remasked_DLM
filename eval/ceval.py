import argparse
import json
import os
import remask.env  # noqa: F401
import time

from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from tqdm import tqdm

from remask import load_remask_model, load_original_model
from remask.utils import format_chat_prompt, tokenize_prompt, extract_choice_answer
from eval.common import add_parallel_args, shard_dataset

CEVAL_PROMPT = (
    '以下是一道选择题，请逐步分析后给出答案，格式为"答案是(X)"。\n\n'
    "问题：{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n答案："
)


def run_tag(args):
    if args.mode == "original":
        return "original"
    return f"remask_{args.strategy}_{args.remask_threshold}"


def load_ceval_val():
    try:
        return load_dataset("ceval/ceval-exam", "all", split="val")
    except Exception:
        names = [n for n in get_dataset_config_names("ceval/ceval-exam") if n != "all"]
        parts = []
        for n in names:
            try:
                parts.append(load_dataset("ceval/ceval-exam", n, split="val"))
            except Exception:
                continue
        if not parts:
            raise
        return concatenate_datasets(parts)


def norm_ceval_answer(a):
    if isinstance(a, int):
        return "ABCD"[a] if 0 <= a < 4 else str(a)
    s = str(a).strip().upper()
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

    dataset = load_ceval_val()
    print(f"C-Eval val: {len(dataset)} items")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = shard_dataset(dataset, args)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_sfx = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.output_dir, f"{tag}{shard_sfx}_results.jsonl")
    correct = total = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(tqdm(dataset, desc=f"C-Eval [{tag}]")):
        gold = norm_ceval_answer(ex["answer"])
        user = CEVAL_PROMPT.format(
            question=ex["question"], A=ex["A"], B=ex["B"], C=ex["C"], D=ex["D"])
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
            question=ex["question"]))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] acc={correct}/{total}={correct/total:.4f}")

    elapsed = time.time() - t0
    acc = correct / total if total else 0
    print(f"\nC-Eval [{tag}] {correct}/{total} = {acc:.4f}  ({elapsed:.0f}s)")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.output_dir, f"{tag}{shard_sfx}_summary.json"), "w") as f:
        json.dump(dict(
            benchmark="ceval", tag=tag, mode=args.mode,
            strategy=args.strategy, remask_threshold=args.remask_threshold,
            accuracy=acc, correct=correct, total=total, time_s=elapsed), f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/ceval")
    p.add_argument("--gen_length", type=int, default=512)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    add_parallel_args(p)
    run(p.parse_args())

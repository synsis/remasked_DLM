"""
python -m eval.drop --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.drop --mode original --batch_size 128
"""

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
from eval.common import (
    add_parallel_args, shard_dataset, _attach_gen_stats,
    aggregate_gen_stats, gen_params_dict, gen_kwargs, get_gen_stats,
)

PROMPT_TPL = (
    "Read the passage and answer the question. "
    "Give only the final answer as briefly as possible (a number, name, or short phrase). "
    'Use the format "The answer is <answer>".\n\n'
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


def _eval_one(resp, ex):
    gold = gold_spans(ex)
    pred = extract_short_answer(resp)
    em_val = max_metric_over_answers(pred, gold, compute_em) if gold else 0.0
    f1_val = max_metric_over_answers(pred, gold, compute_f1) if gold else 0.0
    return dict(
        passage=ex["passage"], question=ex["question"], gold=gold,
        predicted=pred, em=em_val, f1=f1_val, correct=(em_val >= 1.0 - 1e-9),
    )


def _flush(summary_path, tag, args, results, t0, batch_size, done=False):
    total = len(results)
    elapsed = time.time() - t0
    avg_em = sum(r["em"] for r in results) / total if total else 0.0
    avg_f1 = sum(r["f1"] for r in results) / total if total else 0.0
    em_correct = sum(1 for r in results if r["em"] >= 1.0 - 1e-9)
    gen_agg = aggregate_gen_stats(results)
    with open(summary_path, "w") as f:
        summary = dict(
            benchmark="drop", tag=tag, mode=args.mode,
            main_metric="f1", avg_em=avg_em, avg_f1=avg_f1,
            accuracy=em_correct / total if total else 0.0,
            correct=em_correct, total=total,
            time_s=elapsed, done=done,
        )
        summary.update(gen_params_dict(args))
        summary.update(gen_agg)
        json.dump(summary, f, indent=2)


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
    dataset = shard_dataset(dataset, args)
    examples = list(dataset)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_sfx = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.output_dir, f"{tag}{shard_sfx}_results.jsonl")
    summary_path = os.path.join(args.output_dir, f"{tag}{shard_sfx}_summary.json")
    results = []
    t0 = time.time()
    gkw = gen_kwargs(args, mask_id)
    batch_size = args.batch_size

    fout = open(out_path, "w")
    total = 0

    if batch_size > 1:
        for start in tqdm(range(0, len(examples), batch_size),
                          desc=f"DROP [{tag}] bs={batch_size}"):
            batch_ex = examples[start : start + batch_size]
            prompts_ids = []
            for ex in batch_ex:
                prompt = format_chat_prompt(
                    PROMPT_TPL.format(passage=ex["passage"], question=ex["question"]),
                    tokenizer)
                prompts_ids.append(tokenize_prompt(prompt, tokenizer, model.device))

            outputs = model.generate_batch(inputs_list=prompts_ids, **gkw)
            batch_stats = get_gen_stats(model)

            for out, ex in zip(outputs, batch_ex):
                resp = tokenizer.decode(out[0], skip_special_tokens=True)
                r = _eval_one(resp, ex)
                r["response"] = resp
                if batch_stats:
                    r['_tpf'] = batch_stats.get('tpf')
                    r['_tps'] = batch_stats.get('tps')
                    r['_forward_passes'] = batch_stats.get('forward_passes')
                    r['_output_tokens'] = batch_stats.get('output_tokens')
                results.append(r)
                total += 1
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()

            if total % 50 < batch_size or start + batch_size >= len(examples):
                _flush(summary_path, tag, args, results, t0, batch_size)
    else:
        for i, ex in enumerate(tqdm(examples, desc=f"DROP [{tag}]")):
            prompt = format_chat_prompt(
                PROMPT_TPL.format(passage=ex["passage"], question=ex["question"]),
                tokenizer)
            ids = tokenize_prompt(prompt, tokenizer, model.device)
            out_tensor = model.generate(inputs=ids, **gkw)
            resp = tokenizer.decode(out_tensor[0], skip_special_tokens=True)
            r = _eval_one(resp, ex)
            r["response"] = resp
            _attach_gen_stats(r, model)
            results.append(r)
            total += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            if total % 10 == 0 or total == len(examples):
                _flush(summary_path, tag, args, results, t0, 1)
            if total % 50 == 0:
                avg_f1 = sum(r2["f1"] for r2 in results) / total
                print(f"  [{total}] F1={avg_f1:.4f}")

    fout.close()
    _flush(summary_path, tag, args, results, t0, batch_size, done=True)
    avg_em = sum(r["em"] for r in results) / total if total else 0
    avg_f1 = sum(r["f1"] for r in results) / total if total else 0
    print(f"\nDROP [{tag}] EM={avg_em:.4f} F1={avg_f1:.4f}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/drop")
    p.add_argument("--gen_length", type=int, default=16384)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    add_parallel_args(p)
    run(p.parse_args())

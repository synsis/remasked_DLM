"""Common helpers for batched / sharded evaluation."""

import json
import os
import time
from tqdm import tqdm

from remask.utils import format_chat_prompt, tokenize_prompt


def get_gen_stats(model):
    """Return the last generation stats dict from model, or empty dict."""
    return getattr(model, '_gen_stats', {}) or {}


def _attach_gen_stats(r, model):
    """Attach per-sample generation stats from model._gen_stats to result dict."""
    stats = get_gen_stats(model)
    if stats:
        r['_tpf'] = stats.get('tpf')
        r['_tps'] = stats.get('tps')
        r['_forward_passes'] = stats.get('forward_passes')
        r['_output_tokens'] = stats.get('output_tokens')
        if 'remask_total' in stats:
            r['_remask_total'] = stats['remask_total']


def aggregate_gen_stats(results):
    """Aggregate _tpf/_tps/etc. across results into summary averages."""
    tpfs = [r['_tpf'] for r in results if r.get('_tpf') is not None]
    tpss = [r['_tps'] for r in results if r.get('_tps') is not None]
    fwds = [r['_forward_passes'] for r in results if r.get('_forward_passes') is not None]
    toks = [r['_output_tokens'] for r in results if r.get('_output_tokens') is not None]
    remasks = [r['_remask_total'] for r in results if r.get('_remask_total') is not None]
    agg = {}
    if tpfs:
        agg['avg_tpf'] = sum(tpfs) / len(tpfs)
    if tpss:
        agg['avg_tps'] = sum(tpss) / len(tpss)
    if fwds:
        agg['avg_forward_passes'] = sum(fwds) / len(fwds)
        agg['total_forward_passes'] = sum(fwds)
    if toks:
        agg['avg_output_tokens'] = sum(toks) / len(toks)
        agg['total_output_tokens'] = sum(toks)
    if remasks:
        agg['avg_remask_total'] = sum(remasks) / len(remasks)
        agg['total_remask'] = sum(remasks)
    return agg


def add_parallel_args(parser):
    """Add --batch_size, --shard_id, --num_shards to an argparse parser."""
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Number of prompts per forward pass (batched generation)")
    parser.add_argument(
        "--shard_id", type=int, default=0,
        help="Dataset shard index for multi-process parallelism")
    parser.add_argument(
        "--num_shards", type=int, default=1,
        help="Total number of dataset shards")
    return parser


def shard_dataset(dataset, args):
    """Shard a HF dataset according to args.shard_id / args.num_shards."""
    if args.num_shards > 1:
        indices = list(range(args.shard_id, len(dataset), args.num_shards))
        dataset = dataset.select(indices)
        print(f"  shard {args.shard_id}/{args.num_shards}: {len(dataset)} samples")
    return dataset


def output_paths(output_dir, tag, args):
    """Return (results_jsonl_path, summary_json_path) with shard suffix if needed."""
    shard_suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    results_path = os.path.join(output_dir, f"{tag}{shard_suffix}_results.jsonl")
    summary_path = os.path.join(output_dir, f"{tag}{shard_suffix}_summary.json")
    return results_path, summary_path


def gen_kwargs(args, mask_id):
    """Build generation keyword args from argparse namespace."""
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


def run_eval(
    model, tokenizer, mask_id, examples, args, tag, benchmark,
    format_prompt_fn, evaluate_fn, make_result_fn=None,
):
    """Generic eval loop with batch + shard support.

    Args:
        model, tokenizer, mask_id: loaded model assets
        examples: list of dicts (dataset examples)
        args: parsed argparse namespace (must include batch_size, shard_id, num_shards,
              gen_length, block_length, steps, threshold, editing_threshold, temperature)
        tag: string tag for output files
        benchmark: benchmark name string
        format_prompt_fn(ex, tokenizer) -> str: produce the user prompt text
        evaluate_fn(response_text, ex) -> dict with at least {"correct": bool}:
            evaluate one response, returns fields to merge into result
        make_result_fn(ex, eval_result, response) -> dict (optional):
            build full result dict; if None, uses eval_result + {"response": response}
    """
    os.makedirs(args.output_dir, exist_ok=True)
    results_path, summary_path = output_paths(args.output_dir, tag, args)

    gkw = gen_kwargs(args, mask_id)
    results = []
    t0 = time.time()
    batch_size = args.batch_size

    if batch_size > 1:
        _run_batched(
            model, tokenizer, examples, batch_size, gkw, results,
            format_prompt_fn, evaluate_fn, make_result_fn, tag, benchmark,
        )
    else:
        _run_sequential(
            model, tokenizer, examples, gkw, results,
            format_prompt_fn, evaluate_fn, make_result_fn, tag, benchmark,
        )

    correct = sum(1 for r in results if r.get("correct"))
    total = len(results)
    elapsed = time.time() - t0
    acc = correct / total if total else 0
    print(f"\n{benchmark} [{tag}] {correct}/{total} = {acc:.4f}  ({elapsed:.0f}s)")

    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    gen_agg = aggregate_gen_stats(results)
    with open(summary_path, "w") as f:
        summary = dict(
            benchmark=benchmark, tag=tag, mode=getattr(args, "mode", None),
            strategy=getattr(args, "strategy", None),
            remask_threshold=getattr(args, "remask_threshold", None),
            accuracy=acc, correct=correct, total=total,
            time_s=elapsed, batch_size=batch_size,
        )
        summary.update(gen_agg)
        json.dump(summary, f, indent=2)
    return results, acc


def _process_one(model, tokenizer, ex, gkw, format_prompt_fn, evaluate_fn, make_result_fn):
    prompt_text = format_prompt_fn(ex, tokenizer)
    ids = tokenize_prompt(prompt_text, tokenizer, model.device)
    out = model.generate(inputs=ids, **gkw)
    resp = tokenizer.decode(out[0], skip_special_tokens=True)
    ev = evaluate_fn(resp, ex)
    if make_result_fn:
        r = make_result_fn(ex, ev, resp)
    else:
        r = {**ev, "response": resp}
    _attach_gen_stats(r, model)
    return r


def _run_sequential(model, tokenizer, examples, gkw, results,
                    format_prompt_fn, evaluate_fn, make_result_fn, tag, benchmark):
    correct = total = 0
    for i, ex in enumerate(tqdm(examples, desc=f"{benchmark} [{tag}]")):
        r = _process_one(model, tokenizer, ex, gkw,
                         format_prompt_fn, evaluate_fn, make_result_fn)
        correct += r.get("correct", False)
        total += 1
        results.append(r)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] acc={correct}/{total}={correct/total:.4f}")


def _run_batched(model, tokenizer, examples, batch_size, gkw, results,
                 format_prompt_fn, evaluate_fn, make_result_fn, tag, benchmark):
    correct = total = 0
    for start in tqdm(range(0, len(examples), batch_size),
                      desc=f"{benchmark} [{tag}] bs={batch_size}"):
        batch_ex = examples[start : start + batch_size]
        prompts_ids = []
        for ex in batch_ex:
            prompt_text = format_prompt_fn(ex, tokenizer)
            prompts_ids.append(tokenize_prompt(prompt_text, tokenizer, model.device))

        outputs = model.generate_batch(inputs_list=prompts_ids, **gkw)
        batch_stats = get_gen_stats(model)

        for out, ex in zip(outputs, batch_ex):
            resp = tokenizer.decode(out[0], skip_special_tokens=True)
            ev = evaluate_fn(resp, ex)
            r = make_result_fn(ex, ev, resp) if make_result_fn else {**ev, "response": resp}
            if batch_stats:
                r['_tpf'] = batch_stats.get('tpf')
                r['_tps'] = batch_stats.get('tps')
                r['_forward_passes'] = batch_stats.get('forward_passes')
                r['_output_tokens'] = batch_stats.get('output_tokens')
                if 'remask_total' in batch_stats:
                    r['_remask_total'] = batch_stats['remask_total']
            correct += r.get("correct", False)
            total += 1
            results.append(r)

        if total > 0 and total % 50 < batch_size:
            print(f"  [{total}] acc={correct}/{total}={correct/total:.4f}")

"""Common helpers for batched / sharded evaluation."""

import json
import os
import time

import torch
from tqdm import tqdm

from remask.utils import format_chat_prompt, tokenize_prompt


@torch.inference_mode()
def generate_progressive(
    model, inputs, mask_id, max_gen_length=16384, chunk_size=4096, **gen_kwargs
):
    """Progressive-extension generation for diffusion models.

    Instead of allocating max_gen_length mask tokens upfront (each forward pass
    attends to the full sequence → O(n²) per step), we start with chunk_size
    tokens and extend only if EOS is not found.

    Mathematically equivalent to a single gen_length=max_gen_length call because
    block-causal attention prevents later mask blocks from influencing earlier
    blocks.  But forward passes operate on shorter sequences, yielding:
      - ~(max/chunk)² less attention compute for problems that finish early
      - ~(max/chunk)² less memory for the block-causal attention mask tensor

    Returns: tensor of shape [1, total_generated_tokens] (same as model.generate).
    """
    gen_kwargs["mask_id"] = mask_id
    gen_kwargs["eos_early_stop"] = True
    EOS_ID = 156892

    all_chunks = []
    current_input = inputs
    remaining = max_gen_length
    total_fwd = 0
    t0 = time.time()

    while remaining > 0:
        this_len = min(chunk_size, remaining)
        out = model.generate(inputs=current_input, gen_length=this_len, **gen_kwargs)

        stats = getattr(model, "_gen_stats", {})
        total_fwd += stats.get("forward_passes", 0)
        out_tokens = stats.get("output_tokens", 0)

        all_chunks.append(out)

        if out_tokens < this_len or (out[0] == EOS_ID).any():
            break

        remaining -= this_len
        current_input = torch.cat([inputs] + all_chunks, dim=1)

    result = torch.cat(all_chunks, dim=1)
    elapsed = time.time() - t0
    total_tok = result.shape[1]
    model._gen_stats = {
        "forward_passes": total_fwd,
        "output_tokens": total_tok,
        "tpf": total_tok / max(1, total_fwd),
        "wall_time_s": elapsed,
        "tps": total_tok / max(1e-9, elapsed),
    }
    return result


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
        if 't2t_edits' in stats:
            r['_t2t_edits'] = stats['t2t_edits']


def aggregate_gen_stats(results):
    """Aggregate _tpf/_tps/etc. across results into summary averages."""
    tpfs = [r['_tpf'] for r in results if r.get('_tpf') is not None]
    tpss = [r['_tps'] for r in results if r.get('_tps') is not None]
    fwds = [r['_forward_passes'] for r in results if r.get('_forward_passes') is not None]
    toks = [r['_output_tokens'] for r in results if r.get('_output_tokens') is not None]
    remasks = [r['_remask_total'] for r in results if r.get('_remask_total') is not None]
    t2t_edits = [r['_t2t_edits'] for r in results if r.get('_t2t_edits') is not None]
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
    if t2t_edits:
        agg['avg_t2t_edits'] = sum(t2t_edits) / len(t2t_edits)
        agg['total_t2t_edits'] = sum(t2t_edits)
    return agg


def add_parallel_args(parser):
    """Add --batch_size, --shard_id, --num_shards, --max_seq_length to an argparse parser."""
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Number of prompts per forward pass (batched generation)")
    parser.add_argument(
        "--shard_id", type=int, default=0,
        help="Dataset shard index for multi-process parallelism")
    parser.add_argument(
        "--num_shards", type=int, default=1,
        help="Total number of dataset shards")
    parser.add_argument(
        "--max_seq_length", type=int, default=None,
        help="Max total seq length (prompt+gen). Overrides gen_length dynamically per batch.")
    return parser


def shard_dataset(dataset, args):
    """Shard a HF dataset according to args.shard_id / args.num_shards."""
    if args.num_shards > 1:
        indices = list(range(args.shard_id, len(dataset), args.num_shards))
        dataset = dataset.select(indices)
        print(f"  shard {args.shard_id}/{args.num_shards}: {len(dataset)} samples")
    return dataset


def output_paths(output_dir, tag, args):
    """Return (results_jsonl_path, summary_json_path) with bsz and shard in filename."""
    bsz = getattr(args, "batch_size", 1)
    bsz_suffix = f"_bsz{bsz}"
    shard_suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    results_path = os.path.join(output_dir, f"{tag}{bsz_suffix}{shard_suffix}_results.jsonl")
    summary_path = os.path.join(output_dir, f"{tag}{bsz_suffix}{shard_suffix}_summary.json")
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


def gen_params_dict(args):
    """Extract all generation / eval hyper-params from argparse namespace."""
    d = {}
    for key in ("gen_length", "block_length", "steps", "threshold",
                "editing_threshold", "temperature", "batch_size",
                "strategy", "remask_threshold", "max_remask_ratio"):
        val = getattr(args, key, None)
        if val is not None:
            d[key] = val
    return d


def _flush_summary(summary_path, benchmark, tag, args, results, t0, batch_size, done=False):
    """Write/update summary JSON incrementally."""
    correct = sum(1 for r in results if r.get("correct"))
    total = len(results)
    elapsed = time.time() - t0
    acc = correct / total if total else 0
    gen_agg = aggregate_gen_stats(results)
    with open(summary_path, "w") as f:
        summary = dict(
            benchmark=benchmark, tag=tag, mode=getattr(args, "mode", None),
            accuracy=acc, correct=correct, total=total,
            time_s=elapsed, done=done,
        )
        summary.update(gen_params_dict(args))
        summary.update(gen_agg)
        json.dump(summary, f, indent=2)
    return correct, total, acc


def run_eval(
    model, tokenizer, mask_id, examples, args, tag, benchmark,
    format_prompt_fn, evaluate_fn, make_result_fn=None,
):
    """Generic eval loop with batch + shard support.

    Results are streamed to JSONL (one line per sample) so that partial
    progress survives crashes.  Summary JSON is refreshed every 10 samples.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    results_path, summary_path = output_paths(args.output_dir, tag, args)

    gkw = gen_kwargs(args, mask_id)
    results = []
    t0 = time.time()
    batch_size = args.batch_size

    fout = open(results_path, "w")
    if batch_size > 1:
        _run_batched(
            model, tokenizer, examples, batch_size, gkw, results,
            format_prompt_fn, evaluate_fn, make_result_fn, tag, benchmark,
            fout=fout, summary_path=summary_path, args=args, t0=t0,
        )
    else:
        _run_sequential(
            model, tokenizer, examples, gkw, results,
            format_prompt_fn, evaluate_fn, make_result_fn, tag, benchmark,
            fout=fout, summary_path=summary_path, args=args, t0=t0,
        )
    fout.close()

    correct, total, acc = _flush_summary(
        summary_path, benchmark, tag, args, results, t0, batch_size, done=True,
    )
    print(f"\n{benchmark} [{tag}] {correct}/{total} = {acc:.4f}  ({time.time()-t0:.0f}s)")
    return results, acc


def _cap_gen_length(gkw, prompt_len, args):
    """If max_seq_length is set, cap gen_length so total doesn't exceed it."""
    max_seq = getattr(args, "max_seq_length", None)
    if max_seq and max_seq > prompt_len:
        capped = max(1, max_seq - prompt_len)
        if capped < gkw.get("gen_length", 0):
            return {**gkw, "gen_length": capped}
    return gkw


def _process_one(model, tokenizer, ex, gkw, format_prompt_fn, evaluate_fn, make_result_fn, args=None):
    prompt_text = format_prompt_fn(ex, tokenizer)
    ids = tokenize_prompt(prompt_text, tokenizer, model.device)
    this_gkw = _cap_gen_length(gkw, ids.shape[1], args) if args else gkw
    out = model.generate(inputs=ids, **this_gkw)
    resp = tokenizer.decode(out[0], skip_special_tokens=True)
    ev = evaluate_fn(resp, ex)
    if make_result_fn:
        r = make_result_fn(ex, ev, resp)
    else:
        r = {**ev, "response": resp}
    _attach_gen_stats(r, model)
    return r


def _run_sequential(model, tokenizer, examples, gkw, results,
                    format_prompt_fn, evaluate_fn, make_result_fn, tag, benchmark,
                    fout=None, summary_path=None, args=None, t0=None):
    correct = total = 0
    for i, ex in enumerate(tqdm(examples, desc=f"{benchmark} [{tag}]")):
        r = _process_one(model, tokenizer, ex, gkw,
                         format_prompt_fn, evaluate_fn, make_result_fn, args=args)
        correct += r.get("correct", False)
        total += 1
        results.append(r)
        if fout is not None:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
        if summary_path and t0 and (total % 10 == 0 or total == len(examples)):
            _flush_summary(summary_path, benchmark, tag, args, results, t0, 1)
        if total % 50 == 0:
            print(f"  [{total}] acc={correct}/{total}={correct/total:.4f}")


def _run_batched(model, tokenizer, examples, batch_size, gkw, results,
                 format_prompt_fn, evaluate_fn, make_result_fn, tag, benchmark,
                 fout=None, summary_path=None, args=None, t0=None):
    correct = total = 0
    for start in tqdm(range(0, len(examples), batch_size),
                      desc=f"{benchmark} [{tag}] bs={batch_size}"):
        batch_ex = examples[start : start + batch_size]
        prompts_ids = []
        for ex in batch_ex:
            prompt_text = format_prompt_fn(ex, tokenizer)
            prompts_ids.append(tokenize_prompt(prompt_text, tokenizer, model.device))

        max_prompt_len = max(p.shape[1] for p in prompts_ids)
        batch_gkw = _cap_gen_length(gkw, max_prompt_len, args)
        total_seq = max_prompt_len + batch_gkw.get("gen_length", 0)
        print(f"  batch {start//batch_size}: prompt={max_prompt_len} gen={batch_gkw.get('gen_length',0)} total_seq={total_seq} bs={len(prompts_ids)}", flush=True)
        outputs = model.generate_batch(inputs_list=prompts_ids, **batch_gkw)
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
            if fout is not None:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()

        if summary_path and t0 and (total % 10 == 0 or start + batch_size >= len(examples)):
            _flush_summary(summary_path, benchmark, tag, args, results, t0, batch_size)
        if total > 0 and total % 50 < batch_size:
            print(f"  [{total}] acc={correct}/{total}={correct/total:.4f}")

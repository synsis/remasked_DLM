"""
python -m eval.humaneval --mode remask --strategy low_prob --remask_threshold 0.1
# then: evalplus.evaluate --dataset humaneval --samples results/humaneval/<tag>_samples.jsonl
"""

import argparse
import json
import os
import time

import torch
from tqdm import tqdm

from remask import load_remask_model, load_original_model
from remask.utils import format_chat_prompt, tokenize_prompt, extract_code_block
from eval.common import add_parallel_args, shard_dataset, output_paths, gen_params_dict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

PROMPT_TPL = (
    "Complete the following Python function. "
    "Only output the complete function implementation, no explanations.\n\n{prompt}"
)


def load_humaneval():
    path = os.path.join(DATA_DIR, "HumanEvalPlus.jsonl")
    if os.path.exists(path):
        problems = {}
        with open(path) as f:
            for line in f:
                p = json.loads(line)
                problems[p["task_id"]] = p
        return problems
    from evalplus.data import get_human_eval_plus
    return get_human_eval_plus()


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
            remask_threshold=args.remask_threshold,
            max_remask_per_pos=getattr(args, "max_remask_per_pos", 3),
            max_remask_ratio=getattr(args, "max_remask_ratio", 0.25))

    problems = load_humaneval()
    print(f"HumanEval+: {len(problems)} problems")
    if args.num_shards > 1:
        keys = [k for i, k in enumerate(problems) if i % args.num_shards == args.shard_id]
        problems = {k: problems[k] for k in keys}
        print(f"  shard {args.shard_id}/{args.num_shards}: {len(problems)} problems")

    os.makedirs(args.output_dir, exist_ok=True)
    shard_sfx = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    samples_path = os.path.join(args.output_dir, f"{tag}{shard_sfx}_samples.jsonl")
    results_path, summary_path = output_paths(args.output_dir, tag, args)

    results = []
    samples = []
    t0 = time.time()

    for tid, prob in tqdm(problems.items(), desc=f"HumanEval+ [{tag}]"):
        if args.instruct:
            prompt = format_chat_prompt(PROMPT_TPL.format(prompt=prob["prompt"]), tokenizer)
        else:
            prompt = prob["prompt"]
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(
            inputs=ids, gen_length=args.gen_length, block_length=args.block_length,
            steps=args.steps, threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            temperature=args.temperature, mask_id=mask_id, eos_early_stop=True,
        )
        comp = tokenizer.decode(out[0], skip_special_tokens=True)
        sol = extract_code_block(comp) if args.instruct else prob["prompt"] + comp
        samples.append({"task_id": tid, "solution": sol})
        results.append({"task_id": tid, "solution": sol, "correct": None})

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s ({elapsed/len(results):.1f}s/prob)")

    with open(samples_path, "w") as f:
        for r in samples:
            f.write(json.dumps(r) + "\n")
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = dict(
        benchmark="humaneval", tag=tag, mode=args.mode,
        total=len(results), time_s=elapsed,
        done=True, needs_evalplus=True,
        **gen_params_dict(args),
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved → {samples_path}")
    print(f"Saved → {results_path}")
    print(f"Evaluate: evalplus.evaluate --dataset humaneval --samples {samples_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/humaneval")
    p.add_argument("--gen_length", type=int, default=16384)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--instruct", action="store_true", default=True)
    p.add_argument("--no_instruct", dest="instruct", action="store_false")
    p.add_argument("--max_remask_per_pos", type=int, default=3)
    p.add_argument("--max_remask_ratio", type=float, default=0.25)
    add_parallel_args(p)
    run(p.parse_args())

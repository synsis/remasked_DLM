"""
Standard MBPP evaluation matching EvalPlus format exactly.

Same prompt as HumanEval (zero-shot, EvalPlus style):
  <user>
  Please provide a self-contained Python script that solves the following
  problem in a markdown code block:
  ```
  {MBPP prompt (docstring with assertions)}
  ```
  </user>
  <assistant>
  Below is a Python script with a self-contained function that solves the
  problem and passes corresponding tests:
  ```python
  {model generates here}

This is the exact same prompt that Meta LLaMA 3.1, Qwen2.5-Coder,
DeepSeek-Coder V2, etc. are evaluated with on the EvalPlus leaderboard.

python -m eval.mbpp_std --mode remask --strategy low_prob --remask_threshold 0.3
# then: evalplus.evaluate --dataset mbpp --samples <samples.jsonl>
"""

import argparse
import json
import os
import time

import torch
from tqdm import tqdm

from remask import load_remask_model, load_original_model
from remask.utils import tokenize_prompt
from eval.common import add_parallel_args, output_paths, gen_params_dict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves "
    "the problem and passes corresponding tests:"
)

_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

CHAT_EOS = [
    "\n```\n",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
]


def _truncate_eos(text):
    """Truncate at the first EOS marker, matching EvalPlus behavior."""
    min_idx = len(text)
    for eos in CHAT_EOS:
        idx = text.find(eos)
        if idx != -1 and idx < min_idx:
            min_idx = idx
    return text[:min_idx].replace("\t", "    ")


def _make_evalplus_prompt(task_prompt, tokenizer):
    """Build the EvalPlus-style chat prompt with assistant prefill."""
    user_msg = f"{INSTRUCTION_PREFIX}\n```\n{task_prompt.strip()}\n```"
    assistant_msg = f"{RESPONSE_PREFIX}\n```python\n{_MAGIC_SPLITTER_}\n```"

    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        tokenize=False,
    )
    prompt = full.split(_MAGIC_SPLITTER_)[0]
    return prompt


def load_mbpp():
    path = os.path.join(DATA_DIR, "MbppPlus.jsonl")
    if os.path.exists(path):
        problems = {}
        with open(path) as f:
            for line in f:
                p = json.loads(line)
                problems[p["task_id"]] = p
        return problems
    from evalplus.data import get_mbpp_plus
    return get_mbpp_plus()


def run_tag(args):
    if args.mode == "original":
        return "original"
    return f"remask_{args.strategy}_{args.remask_threshold}"


def _sanitize_solution(raw_output, entry_point):
    """Apply EvalPlus sanitize if available, else basic extraction."""
    try:
        from evalplus.sanitize import sanitize
        return sanitize(raw_output, entrypoint=entry_point)
    except ImportError:
        pass
    import re
    m = re.search(r"```(?:python)?\s*\n(.*?)```", raw_output, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw_output.strip()


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

    problems = load_mbpp()
    print(f"MBPP+ (EvalPlus std): {len(problems)} problems")
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

    for tid, prob in tqdm(problems.items(), desc=f"MBPP+ std [{tag}]"):
        prompt = _make_evalplus_prompt(prob["prompt"], tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(
            inputs=ids, gen_length=args.gen_length, block_length=args.block_length,
            steps=args.steps, threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            temperature=args.temperature, mask_id=mask_id, eos_early_stop=True,
        )
        raw = tokenizer.decode(out[0], skip_special_tokens=True)
        impl = _truncate_eos(raw)
        sol = _sanitize_solution(impl, prob["entry_point"])

        samples.append({"task_id": tid, "solution": sol})
        results.append({"task_id": tid, "solution": sol, "raw": raw, "correct": None})

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s ({elapsed/len(results):.1f}s/prob)")

    with open(samples_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = dict(
        benchmark="mbpp_std", tag=tag, mode=args.mode,
        total=len(results), time_s=elapsed,
        done=True, needs_evalplus=True,
        prompt_format="evalplus_standard",
        **gen_params_dict(args),
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved → {samples_path}")
    print(f"Evaluate: evalplus.evaluate --dataset mbpp --samples {samples_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/mbpp_std")
    p.add_argument("--gen_length", type=int, default=768)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_remask_per_pos", type=int, default=3)
    p.add_argument("--max_remask_ratio", type=float, default=0.25)
    add_parallel_args(p)
    run(p.parse_args())

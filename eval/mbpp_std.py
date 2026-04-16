"""
Standard 3-shot MBPP evaluation following lm-evaluation-harness format.

Prompt template (per lm-evaluation-harness mbpp.yaml):
  You are an expert Python programmer, and here is your task: {text}
  Your code should pass these tests:

  {test0}
  {test1}
  {test2}
  [BEGIN]
  {code}     <-- only in few-shot examples
  [DONE]     <-- only in few-shot examples

3 few-shot examples (tasks 2, 3, 4) are prepended.

python -m eval.mbpp_std --mode remask --strategy low_prob --remask_threshold 0.3
"""

import argparse
import json
import os
import re
import time

import torch
from tqdm import tqdm

from remask import load_remask_model, load_original_model
from remask.utils import format_chat_prompt, tokenize_prompt, extract_code_block
from eval.common import add_parallel_args, output_paths, gen_params_dict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

FEWSHOT_EXAMPLES = [
    {
        "text": "Write a function to find the similar elements from the given two tuple lists.",
        "code": (
            "def similar_elements(test_tup1, test_tup2):\r\n"
            "  res = tuple(set(test_tup1) & set(test_tup2))\r\n"
            "  return (res) "
        ),
        "test_list": [
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
            "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
        ],
    },
    {
        "text": "Write a python function to identify non-prime numbers.",
        "code": (
            "import math\r\n"
            "def is_not_prime(n):\r\n"
            "    result = False\r\n"
            "    for i in range(2,int(math.sqrt(n)) + 1):\r\n"
            "        if n % i == 0:\r\n"
            "            result = True\r\n"
            "    return result"
        ),
        "test_list": [
            "assert is_not_prime(2) == False",
            "assert is_not_prime(10) == True",
            "assert is_not_prime(35) == True",
        ],
    },
    {
        "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
        "code": (
            "import heapq as hq\r\n"
            "def heap_queue_largest(nums,n):\r\n"
            "  largest_nums = hq.nlargest(n, nums)\r\n"
            "  return largest_nums"
        ),
        "test_list": [
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
        ],
    },
]


def _format_one(text, test_list):
    tests = "\n".join(test_list)
    return (
        f"You are an expert Python programmer, and here is your task: {text} "
        f"Your code should pass these tests:\n\n{tests}\n[BEGIN]\n"
    )


def _build_fewshot_prefix():
    parts = []
    for ex in FEWSHOT_EXAMPLES:
        prompt = _format_one(ex["text"], ex["test_list"])
        parts.append(f"{prompt}{ex['code']}\n[DONE]\n\n")
    return "".join(parts)


FEWSHOT_PREFIX = _build_fewshot_prefix()


def _parse_prompt_field(prompt_str):
    """Parse MbppPlus `prompt` docstring into (text, test_list)."""
    body = prompt_str.strip().strip('"""').strip()
    lines = body.split("\n")
    text_lines, tests = [], []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("assert "):
            tests.append(stripped)
        else:
            if not tests:
                text_lines.append(stripped)
    text = " ".join(t for t in text_lines if t)
    return text, tests


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


def _extract_solution(full_output):
    """Extract code from the model response.

    Priority:
    1. [BEGIN]...[DONE] markers (lm-eval-harness style)
    2. Markdown ```python ... ``` code block
    3. Raw output as fallback
    """
    begin_idx = full_output.rfind("[BEGIN]")
    if begin_idx != -1:
        code_start = begin_idx + len("[BEGIN]")
        done_idx = full_output.find("[DONE]", code_start)
        if done_idx == -1:
            return full_output[code_start:].strip()
        return full_output[code_start:done_idx].strip()

    code = extract_code_block(full_output)
    if code != full_output:
        return code

    return full_output.strip()


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
    print(f"MBPP+ (std 3-shot): {len(problems)} problems")

    fewshot_ids = {"Mbpp/2", "Mbpp/3", "Mbpp/4"}
    problems = {k: v for k, v in problems.items() if k not in fewshot_ids}
    print(f"  after excluding few-shot examples: {len(problems)} problems")

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
        text, test_list = _parse_prompt_field(prob["prompt"])
        if not test_list:
            assertion = prob.get("assertion", "")
            test_list = [l.strip() for l in assertion.strip().split("\n") if l.strip().startswith("assert")]
            test_list = test_list[:3]

        query = FEWSHOT_PREFIX + _format_one(text, test_list)
        prompt = format_chat_prompt(query, tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(
            inputs=ids, gen_length=args.gen_length, block_length=args.block_length,
            steps=args.steps, threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            temperature=args.temperature, mask_id=mask_id, eos_early_stop=True,
        )
        comp = tokenizer.decode(out[0], skip_special_tokens=True)
        sol = _extract_solution(comp)

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
        benchmark="mbpp_std", tag=tag, mode=args.mode,
        total=len(results), time_s=elapsed,
        done=True, needs_evalplus=True,
        prompt_format="3-shot lm-eval-harness",
        **gen_params_dict(args),
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved → {samples_path}")
    print(f"Saved → {results_path}")
    print(f"Evaluate: evalplus.evaluate --dataset mbpp --samples {samples_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument("--strategy", choices=["low_prob", "t2t_remask", "logit_diff"], default="low_prob")
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--output_dir", default="results/mbpp_std")
    p.add_argument("--gen_length", type=int, default=16384)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_remask_per_pos", type=int, default=3)
    p.add_argument("--max_remask_ratio", type=float, default=0.25)
    add_parallel_args(p)
    run(p.parse_args())

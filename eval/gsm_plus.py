"""
python -m eval.gsm_plus --mode remask --strategy low_prob --remask_threshold 0.1
python -m eval.gsm_plus --mode original
python -m eval.gsm_plus --mode original --batch_size 4       # batched inference
python -m eval.gsm_plus --mode remask --shard_id 0 --num_shards 2  # data parallel
"""

import argparse
import json
import os
import remask.env  # noqa: F401
import re
import time

import torch
from datasets import load_dataset
from tqdm import tqdm

from remask import load_remask_model, load_original_model
from remask.utils import format_chat_prompt, tokenize_prompt, extract_math_answer, normalize_numeric
from eval.common import _attach_gen_stats, aggregate_gen_stats, get_gen_stats

GSM_PLUS_FEW_SHOT = (
    "Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\n"
    "Let's think step by step\n"
    "Answer: Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n\n"
    "Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\n"
    "Let's think step by step\n"
    "Answer: Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe answer is 201\n\n"
    "Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\n"
    "Let's think step by step\n"
    "Answer: When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe answer is 140\n\n"
    "Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\n"
    "Let's think step by step\n"
    "Answer: For the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe answer is 146\n\n"
    "Question: {question}\n"
    "Let's think step by step\n"
    "Answer:"
)


def extract_gold(answer_text):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", str(answer_text))
    if m:
        return m.group(1).replace(",", "").strip()
    return str(answer_text).replace(",", "").strip()


def run_tag(args):
    if args.mode == "original":
        return "original"
    tag = f"remask_{args.strategy}_{args.remask_threshold}"
    if hasattr(args, 'max_remask_ratio') and args.max_remask_ratio != 0.25:
        tag += f"_r{args.max_remask_ratio}"
    return tag


def gen_kwargs(args, mask_id):
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


def _flush_summary(summary_path, tag, args, results, t0, batch_size, done=False):
    correct = sum(1 for r in results if r.get("correct"))
    total = len(results)
    elapsed = time.time() - t0
    acc = correct / total if total else 0
    gen_agg = aggregate_gen_stats(results)
    with open(summary_path, "w") as f:
        summary = dict(
            benchmark="gsm_plus", tag=tag, mode=args.mode,
            accuracy=acc, correct=correct, total=total, time_s=elapsed,
            strategy=args.strategy, remask_threshold=args.remask_threshold,
            max_remask_ratio=getattr(args, 'max_remask_ratio', None),
            batch_size=batch_size, done=done,
        )
        summary.update(gen_agg)
        json.dump(summary, f, indent=2)


def run(args):
    tag = run_tag(args)

    if args.mode == "original":
        model, tokenizer, mask_id = load_original_model(args.model_path)
    else:
        model, tokenizer, mask_id = load_remask_model(
            args.model_path,
            strategy=args.strategy,
            remask_threshold=args.remask_threshold,
            max_remask_ratio=args.max_remask_ratio,
        )

    try:
        dataset = load_dataset("qintongli/GSM-Plus", split="testmini")
    except Exception:
        dataset = load_dataset("qintongli/GSM-Plus", split="test")
    print(f"GSM-Plus: {len(dataset)} problems")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    if args.num_shards > 1:
        indices = list(range(args.shard_id, len(dataset), args.num_shards))
        dataset = dataset.select(indices)
        print(f"  shard {args.shard_id}/{args.num_shards}: {len(dataset)} samples")

    os.makedirs(args.output_dir, exist_ok=True)
    shard_suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.output_dir, f"{tag}{shard_suffix}_results.jsonl")
    summary_path = os.path.join(args.output_dir, f"{tag}{shard_suffix}_summary.json")

    correct = total = 0
    results = []
    t0 = time.time()
    gkw = gen_kwargs(args, mask_id)
    batch_size = args.batch_size

    fout = open(out_path, "w")
    if batch_size > 1:
        _run_batched(model, tokenizer, dataset, batch_size, gkw, results, args, tag,
                     fout=fout, summary_path=summary_path, t0=t0)
    else:
        _run_sequential(model, tokenizer, dataset, gkw, results, args, tag,
                        fout=fout, summary_path=summary_path, t0=t0)
    fout.close()

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    elapsed = time.time() - t0
    acc = correct / total if total else 0
    print(f"\nGSM-Plus [{tag}] {correct}/{total} = {acc:.4f}  ({elapsed:.0f}s)")

    _flush_summary(summary_path, tag, args, results, t0, batch_size, done=True)


def _run_sequential(model, tokenizer, dataset, gkw, results, args, tag,
                    fout=None, summary_path=None, t0=None):
    correct = total = 0
    for i, ex in enumerate(tqdm(dataset, desc=f"GSM-Plus [{tag}]")):
        gold = extract_gold(ex["answer"])
        prompt = format_chat_prompt(
            GSM_PLUS_FEW_SHOT.format(question=ex["question"]), tokenizer
        )
        ids = tokenize_prompt(prompt, tokenizer, model.device)

        out = model.generate(inputs=ids, **gkw)
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = extract_math_answer(resp)
        ok = normalize_numeric(pred) == normalize_numeric(gold)
        correct += ok
        total += 1
        r = dict(
            question=ex["question"],
            gold=gold,
            predicted=pred,
            correct=ok,
            response=resp,
        )
        _attach_gen_stats(r, model)
        results.append(r)
        if fout is not None:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
        if summary_path and t0 and (total % 10 == 0 or total == len(dataset)):
            _flush_summary(summary_path, tag, args, results, t0, 1)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] acc={correct}/{total}={correct/total:.4f}")


def _run_batched(model, tokenizer, dataset, batch_size, gkw, results, args, tag,
                 fout=None, summary_path=None, t0=None):
    correct = total = 0
    examples = list(dataset)
    for start in tqdm(range(0, len(examples), batch_size),
                      desc=f"GSM-Plus [{tag}] bs={batch_size}"):
        batch_ex = examples[start : start + batch_size]
        golds, prompts_ids = [], []
        for ex in batch_ex:
            golds.append(extract_gold(ex["answer"]))
            prompt = format_chat_prompt(
                GSM_PLUS_FEW_SHOT.format(question=ex["question"]), tokenizer
            )
            prompts_ids.append(tokenize_prompt(prompt, tokenizer, model.device))

        outputs = model.generate_batch(inputs_list=prompts_ids, **gkw)
        batch_stats = get_gen_stats(model)

        for j, (out, gold, ex) in enumerate(zip(outputs, golds, batch_ex)):
            resp = tokenizer.decode(out[0], skip_special_tokens=True)
            pred = extract_math_answer(resp)
            ok = normalize_numeric(pred) == normalize_numeric(gold)
            correct += ok
            total += 1
            r = dict(
                question=ex["question"],
                gold=gold,
                predicted=pred,
                correct=ok,
                response=resp,
            )
            if batch_stats:
                r['_tpf'] = batch_stats.get('tpf')
                r['_tps'] = batch_stats.get('tps')
                r['_forward_passes'] = batch_stats.get('forward_passes')
                r['_output_tokens'] = batch_stats.get('output_tokens')
                if 'remask_total' in batch_stats:
                    r['_remask_total'] = batch_stats['remask_total']
            results.append(r)
            if fout is not None:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()

        if summary_path and t0 and (total % 10 == 0 or start + batch_size >= len(examples)):
            _flush_summary(summary_path, tag, args, results, t0, batch_size)
        if total > 0 and total % 50 < batch_size:
            print(f"  [{total}] acc={correct}/{total}={correct/total:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--mode", choices=["original", "remask"], default="remask")
    p.add_argument(
        "--strategy",
        choices=["low_prob", "t2t_remask", "logit_diff"],
        default="low_prob",
    )
    p.add_argument("--remask_threshold", type=float, default=None)
    p.add_argument("--max_remask_ratio", type=float, default=0.25,
                    help="Max fraction of editable tokens to remask per step")
    p.add_argument("--output_dir", default="results/gsm_plus")
    p.add_argument("--gen_length", type=int, default=16384)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    # Parallelism options
    p.add_argument("--batch_size", type=int, default=1,
                    help="Number of prompts to process simultaneously (batched forward pass)")
    p.add_argument("--shard_id", type=int, default=0,
                    help="Dataset shard index for multi-process parallelism")
    p.add_argument("--num_shards", type=int, default=1,
                    help="Total number of shards for multi-process parallelism")
    run(p.parse_args())

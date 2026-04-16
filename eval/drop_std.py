"""DROP with 3-shot (LLaMA 3.1 / lm-evaluation-harness standard).

Gold answers and F1/EM metrics are *exactly* aligned with
lm-evaluation-harness `lm_eval/tasks/drop/utils.py`:
  - Gold set = answer + validated_answers (parsed via parse_answer)
  - F1 uses bag-of-words alignment (_align_bags, _match_numbers_if_present)
  - Normalize = lower → remove punc (keep numbers) → remove articles → strip
Few-shot target = first gold tuple joined with "," (matching doc_to_target).

python -m eval.drop_std --mode remask --strategy low_prob --remask_threshold 0.3
"""

import argparse
import json
import os
import re
import string
import remask.env  # noqa: F401
import time

import numpy as np
from datasets import load_dataset
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from remask import load_remask_model, load_original_model
from remask.utils import (
    format_chat_prompt,
    tokenize_prompt,
    extract_short_answer,
)
from eval.common import (
    add_parallel_args, shard_dataset, _attach_gen_stats,
    aggregate_gen_stats, gen_params_dict, gen_kwargs, get_gen_stats,
)

# ---------------------------------------------------------------------------
# lm-eval-harness DROP metric helpers  (verbatim from lm_eval/tasks/drop/utils.py)
# ---------------------------------------------------------------------------
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def _remove_articles(text):
    return _ARTICLES.sub(" ", text)


def _white_space_fix(text):
    return " ".join(text.split())


def _remove_punc(text):
    exclude = set(string.punctuation)
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in exclude)
    else:
        return text


def _fix_number(text):
    return str(float(text)) if _is_number(text) else text


def _tokenize(text):
    return re.split(" |-", text)


def _normalize(answer):
    tokens = [
        _white_space_fix(_remove_articles(_fix_number(_remove_punc(token.lower()))))
        for token in _tokenize(answer)
    ]
    tokens = [token for token in tokens if token.strip()]
    return " ".join(tokens).strip()


def _answer_to_bags(answer):
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _compute_f1(predicted_bag, gold_bag):
    intersection = len(gold_bag.intersection(predicted_bag))
    precision = 1.0 if not predicted_bag else intersection / float(len(predicted_bag))
    recall = 1.0 if not gold_bag else intersection / float(len(gold_bag))
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _match_numbers_if_present(gold_bag, predicted_bag):
    gold_numbers = {w for w in gold_bag if _is_number(w)}
    predicted_numbers = {w for w in predicted_bag if _is_number(w)}
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def _align_bags(predicted, gold):
    scores = np.zeros([len(gold), len(predicted)])
    for gi, g_item in enumerate(gold):
        for pi, p_item in enumerate(predicted):
            if _match_numbers_if_present(g_item, p_item):
                scores[gi, pi] = _compute_f1(p_item, g_item)
    row_ind, col_ind = linear_sum_assignment(-scores)
    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _drop_get_metrics(predicted, gold):
    """Exact lm-eval-harness get_metrics: returns (em, f1)."""
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    if (set(predicted_bags[0]) == set(gold_bags[0])
            and len(predicted_bags[0]) == len(gold_bags[0])):
        exact_match = 1.0
    else:
        exact_match = 0.0
    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = round(float(np.mean(f1_per_bag)), 2)
    return exact_match, f1


# ---------------------------------------------------------------------------
# lm-eval-harness gold-answer extraction  (verbatim logic)
# ---------------------------------------------------------------------------

def _parse_answer(answer):
    if answer["number"] != "":
        return (str(answer["number"]),)
    if answer["spans"] != []:
        return tuple(answer["spans"])
    return (
        " ".join(
            [answer["date"]["day"], answer["date"]["month"], answer["date"]["year"]]
        ).strip(),
    )


def _flatten_validated_answers(validated_answers):
    valid = []
    for i in range(len(validated_answers["number"])):
        valid.append({
            "number": validated_answers["number"][i],
            "date": validated_answers["date"][i],
            "spans": validated_answers["spans"][i],
        })
    return valid


def _get_all_answers(doc):
    """Replicate lm-eval-harness get_answers: answer + validated_answers."""
    answers = []
    answers_set = set()
    candidates = [doc["answer"]] + _flatten_validated_answers(doc["validated_answers"])
    for candidate in candidates:
        answer = _parse_answer(candidate)
        if answer in answers_set:
            continue
        answers_set.add(answer)
        answers.append(answer)
    return answers


# ---------------------------------------------------------------------------
N_FEWSHOT = 3
_fewshot_prefix = None


def _load_fewshot():
    global _fewshot_prefix
    if _fewshot_prefix is not None:
        return
    ds = load_dataset("EleutherAI/drop", split=f"train[:{N_FEWSHOT}]")
    parts = []
    for ex in ds:
        gold = _get_all_answers(ex)
        ans = ",".join(gold[0]) if gold else ""
        parts.append(f"{ex['passage']} {ex['question']}{ans}")
    _fewshot_prefix = "\n\n".join(parts) + "\n\n"
    print(f"Loaded {N_FEWSHOT} DROP fewshot examples")


def run_tag(args):
    if args.mode == "original":
        return "original"
    return f"remask_{args.strategy}_{args.remask_threshold}"


def _eval_one(resp, ex):
    """Score using lm-eval-harness process_results logic."""
    golds = _get_all_answers(ex)
    pred = extract_short_answer(resp)
    max_em = 0.0
    max_f1 = 0.0
    for gold_answer in golds:
        if gold_answer[0].strip():
            em, f1 = _drop_get_metrics(pred, gold_answer)
            max_em = max(max_em, em)
            max_f1 = max(max_f1, f1)
    gold_display = [",".join(g) for g in golds]
    return dict(
        passage=ex["passage"], question=ex["question"], gold=gold_display,
        predicted=pred, em=max_em, f1=max_f1, correct=(max_em >= 1.0 - 1e-9),
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


def _load_drop_split():
    for name in ["EleutherAI/drop", "drop", "ucinlp/drop"]:
        try:
            return load_dataset(name, split="validation")
        except Exception as e:
            print(f"  load_dataset({name!r}) failed: {e}")
    raise RuntimeError("Could not load DROP validation split")


def run(args):
    _load_fewshot()
    tag = run_tag(args)

    if args.mode == "original":
        model, tokenizer, mask_id = load_original_model(args.model_path)
    else:
        model, tokenizer, mask_id = load_remask_model(
            args.model_path, strategy=args.strategy,
            remask_threshold=args.remask_threshold,
            max_remask_per_pos=getattr(args, "max_remask_per_pos", 3),
            max_remask_ratio=getattr(args, "max_remask_ratio", 0.25))

    dataset = _load_drop_split()
    print(f"DROP: {len(dataset)} examples (3-shot)")
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
    total = 0

    fout = open(out_path, "w")

    if batch_size > 1:
        for start in tqdm(range(0, len(examples), batch_size),
                          desc=f"DROP [{tag}] bs={batch_size}"):
            batch_ex = examples[start : start + batch_size]
            prompts_ids = []
            for ex in batch_ex:
                user_msg = _fewshot_prefix + f"{ex['passage']} {ex['question']}"
                prompt = format_chat_prompt(user_msg, tokenizer)
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
            user_msg = _fewshot_prefix + f"{ex['passage']} {ex['question']}"
            prompt = format_chat_prompt(user_msg, tokenizer)
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
    p.add_argument("--gen_length", type=int, default=256)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--editing_threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_remask_per_pos", type=int, default=3)
    p.add_argument("--max_remask_ratio", type=float, default=0.25)
    add_parallel_args(p)
    run(p.parse_args())

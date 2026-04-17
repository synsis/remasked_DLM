"""Aggregate metrics for all results_std datasets.

Handles:
- BBH: re-score with STRICT bbh_correct (whole-word match) to avoid the
  lenient substring bug (tn in rn).
- MMLU-Pro: sum correct/total from summaries (extract_choice_answer already
  applied at eval time).
- TriviaQA / DROP: sum em/f1 (length-weighted) over summaries.
- HumanEval / MBPP: skipped -- needs external EvalPlus evaluation.
"""

import json, os, glob, re, string, sys

BASE = "/vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask/results_std"
TAG = "lowprob_t0.3_c1_r0.25"

# ---------------------------------------------------------------------------
# BBH strict matching
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())

def bbh_correct_strict(resp: str, target: str) -> bool:
    rn, tn = _norm(resp), _norm(target)
    if not tn:
        return False
    if rn == tn:
        return True
    # Whole-word match instead of raw substring
    return bool(re.search(rf"(?:^|\s){re.escape(tn)}(?:$|\s)", rn))


def aggregate_counts(ds):
    """For BBH/MMLU-Pro: sum correct/total across all shards per mode."""
    dpath = f"{BASE}/{ds}/{TAG}"
    per_mode = {"original": [0, 0], "remask": [0, 0]}  # [correct, total]
    for f in sorted(glob.glob(f"{dpath}/*_summary.json")):
        with open(f) as fh:
            s = json.load(fh)
        mode = "original" if os.path.basename(f).startswith("original") else "remask"
        per_mode[mode][0] += s.get("correct", 0)
        per_mode[mode][1] += s.get("total", 0)
    return per_mode


def rescore_bbh_strict():
    """Re-score BBH using strict whole-word match from _results.jsonl files."""
    dpath = f"{BASE}/bbh/{TAG}"
    per_mode = {"original": [0, 0], "remask": [0, 0]}
    for f in sorted(glob.glob(f"{dpath}/*_results.jsonl")):
        mode = "original" if os.path.basename(f).startswith("original") else "remask"
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                # Use the extracted prediction (short text), NOT the full response,
                # because bbh_correct is called on the extracted answer at eval time.
                resp = rec.get("predicted") or ""
                target = rec.get("target") or rec.get("gold") or ""
                per_mode[mode][1] += 1
                if bbh_correct_strict(resp, target):
                    per_mode[mode][0] += 1
    return per_mode


def aggregate_f1_em(ds):
    """For TriviaQA/DROP: compute sample-weighted avg of em/f1."""
    dpath = f"{BASE}/{ds}/{TAG}"
    per_mode = {"original": [0.0, 0.0, 0], "remask": [0.0, 0.0, 0]}  # [sum_em, sum_f1, n_samples]
    for f in sorted(glob.glob(f"{dpath}/*_summary.json")):
        with open(f) as fh:
            s = json.load(fh)
        mode = "original" if os.path.basename(f).startswith("original") else "remask"
        total = s.get("total", 0)
        em = s.get("avg_em", 0.0) * total
        f1 = s.get("avg_f1", 0.0) * total
        per_mode[mode][0] += em
        per_mode[mode][1] += f1
        per_mode[mode][2] += total
    return per_mode


def pct(a, b):
    return 100.0 * a / b if b else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print(f"{'Dataset':<22} {'Mode':<10} {'Original':<30} {'Remask':<30}")
print("=" * 90)

# BBH - both lenient (summary) and strict (re-scored)
bbh_lenient = aggregate_counts("bbh")
bbh_strict = rescore_bbh_strict()
for label, pm in [("bbh (summary=lenient)", bbh_lenient), ("bbh (rescored=strict)", bbh_strict)]:
    o_c, o_t = pm["original"]
    r_c, r_t = pm["remask"]
    print(f"{label:<22} {'acc':<10} {f'{o_c}/{o_t} = {pct(o_c,o_t):.2f}%':<30} {f'{r_c}/{r_t} = {pct(r_c,r_t):.2f}%':<30}")

# MMLU-Pro - summary is authoritative (extract_choice_answer already fixed)
mmlu = aggregate_counts("mmlu_pro")
o_c, o_t = mmlu["original"]; r_c, r_t = mmlu["remask"]
print(f"{'mmlu_pro':<22} {'acc':<10} {f'{o_c}/{o_t} = {pct(o_c,o_t):.2f}%':<30} {f'{r_c}/{r_t} = {pct(r_c,r_t):.2f}%':<30}")

# TriviaQA
tqa = aggregate_f1_em("triviaqa")
o_em, o_f1, o_n = tqa["original"]; r_em, r_f1, r_n = tqa["remask"]
print(f"{'triviaqa':<22} {'em':<10} {f'{pct(o_em,o_n):.2f}% (n={o_n})':<30} {f'{pct(r_em,r_n):.2f}% (n={r_n})':<30}")
print(f"{'':<22} {'f1':<10} {f'{pct(o_f1,o_n):.2f}%':<30} {f'{pct(r_f1,r_n):.2f}%':<30}")

# DROP
drop = aggregate_f1_em("drop")
o_em, o_f1, o_n = drop["original"]; r_em, r_f1, r_n = drop["remask"]
if o_n or r_n:
    print(f"{'drop':<22} {'em':<10} {f'{pct(o_em,o_n):.2f}% (n={o_n})':<30} {f'{pct(r_em,r_n):.2f}% (n={r_n})':<30}")
    print(f"{'':<22} {'f1':<10} {f'{pct(o_f1,o_n):.2f}%':<30} {f'{pct(r_f1,r_n):.2f}%':<30}")
else:
    print(f"{'drop':<22} not yet ready")

# HumanEval/MBPP (needs evalplus)
for ds in ["humaneval", "mbpp"]:
    dpath = f"{BASE}/{ds}/{TAG}"
    sums = sorted(glob.glob(f"{dpath}/*_summary.json"))
    n_orig = sum(1 for s in sums if os.path.basename(s).startswith("original"))
    n_rem = sum(1 for s in sums if os.path.basename(s).startswith("remask"))
    print(f"{ds:<22} {'(evalplus)':<10} {f'{n_orig} shards done':<30} {f'{n_rem} shards done':<30}")

print("=" * 90)

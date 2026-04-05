"""
Scan DROP examples to find T2T-fail + T2M-correct flip cases.
Single-sample mode, max_remask_per_pos=2.
"""
import json, time, gc, os, sys, torch
import remask.env
from remask.loader import load_original_model, load_remask_model
from remask.utils import format_chat_prompt, tokenize_prompt, extract_short_answer
from remask.utils import compute_f1, compute_em, max_metric_over_answers
from datasets import load_dataset

MASK_ID = 156895

DROP_PROMPT = (
    "Read the passage and answer the question. "
    "Give only the final answer as briefly as possible (a number, name, or short phrase). "
    'Use the format "The answer is <answer>".\n\n'
    "Passage: {passage}\n\nQuestion: {question}\n\nAnswer:"
)


def eval_drop(resp, ex):
    gold = ex.get("answers_spans", {}).get("spans", [])
    pred = extract_short_answer(resp)
    em = max_metric_over_answers(pred, gold, compute_em) if gold else 0.0
    return pred, gold, em >= 1.0 - 1e-9


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    ds = load_dataset("drop", split="validation")
    indices = list(range(start, min(start + count, len(ds))))

    gen_kw = dict(gen_length=512, block_length=32, steps=32,
                  threshold=0.7, editing_threshold=0.5, temperature=0.0,
                  mask_id=MASK_ID, eos_id=156892)

    # T2T pass
    print(f"=== T2T pass ({len(indices)} examples) ===", flush=True)
    model, tokenizer, _ = load_original_model()
    t2t_results = {}
    for i, idx in enumerate(indices):
        ex = ds[idx]
        prompt = format_chat_prompt(
            DROP_PROMPT.format(passage=ex["passage"], question=ex["question"]), tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)
        out = model.generate(inputs=ids, **gen_kw)
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        pred, gold, correct = eval_drop(resp, ex)
        t2t_results[idx] = {"pred": pred, "gold": gold, "correct": correct}
        if (i + 1) % 20 == 0:
            print(f"  T2T {i+1}/{len(indices)}", flush=True)

    del model; gc.collect()
    torch.cuda.empty_cache()

    # T2M pass (only on T2T-wrong examples)
    wrong_indices = [idx for idx in indices if not t2t_results[idx]["correct"]]
    print(f"\n=== T2M pass ({len(wrong_indices)} T2T-wrong examples) ===", flush=True)
    model, tokenizer, _ = load_remask_model(
        strategy="low_prob", remask_threshold=None,
        max_remask_per_pos=2, max_remask_ratio=0.25)
    flips = []
    for i, idx in enumerate(wrong_indices):
        ex = ds[idx]
        prompt = format_chat_prompt(
            DROP_PROMPT.format(passage=ex["passage"], question=ex["question"]), tokenizer)
        ids = tokenize_prompt(prompt, tokenizer, model.device)
        out = model.generate(inputs=ids, **gen_kw)
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        pred, gold, correct = eval_drop(resp, ex)
        if correct:
            flips.append({
                "idx": idx, "question": ex["question"],
                "gold": gold, "t2t_pred": t2t_results[idx]["pred"], "t2m_pred": pred,
            })
            print(f"  *** FLIP [{idx}] gold={gold} t2t={t2t_results[idx]['pred']} t2m={pred}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  T2M {i+1}/{len(wrong_indices)}", flush=True)

    print(f"\n=== Found {len(flips)} flips in {len(indices)} examples ===")
    for f in flips:
        print(f"  [{f['idx']}] gold={f['gold']} t2t={f['t2t_pred']} t2m={f['t2m_pred']}  Q: {f['question'][:60]}")

    with open(f"results_v2/flip_scan_{start}_{start+count}.json", "w") as fout:
        json.dump(flips, fout, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

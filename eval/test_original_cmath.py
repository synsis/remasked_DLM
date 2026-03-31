"""Test CMATH using the ORIGINAL LLaDA2.1-mini code (no modifications).
Usage: CUDA_VISIBLE_DEVICES=6 conda run -n remask python test_original_cmath.py
"""
import json, os, re, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_PATH = "/vepfs-mlp2/c20250506/251105017/yaolin/LLaDA2.1-mini"
OUT_DIR = "results/cmath_original_raw"
os.makedirs(OUT_DIR, exist_ok=True)

GEN_KW = dict(
    gen_length=16384,
    block_length=32,
    steps=32,
    threshold=0.7,
    editing_threshold=0.5,
    temperature=0.0,
    eos_early_stop=True,
)

PROMPT = (
    "请逐步解决以下数学问题。\n\n"
    "问题：{question}\n\n"
    "请一步一步思考，最后一行请用「答案是X」的格式给出最终数字答案。"
)

def extract_answer(text):
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m: return m.group(1).strip()
    m = re.search(r'[Tt]he answer is[:\s]*(-?[\d,]+(?:\.\d+)?)', text)
    if m: return m.group(1).replace(",", "").strip()
    m = re.search(r'[Aa]nswer:\s*\$?\\?(?:boxed\{)?(-?[\d,]+(?:\.\d+)?)\}?\$?', text)
    if m: return m.group(1).replace(",", "").strip()
    m = re.search(r'答案[是为：:\s]+(-?[\d,]+(?:\.\d+)?)', text)
    if m: return m.group(1).replace(",", "").strip()
    m = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if m: return m.group(1).replace(",", "").strip()
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    return nums[-1] if nums else None

def normalize(s):
    if s is None: return None
    s = str(s).strip().replace(",", "")
    try: return str(int(float(s))) if float(s) == int(float(s)) else s
    except: return s

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, device_map="auto",
)
model = model.to(torch.bfloat16).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
mask_id = tokenizer.convert_tokens_to_ids("<|mask|>")

print("Loading CMATH...")
for path in ("weitianwen/cmath", "CMATH/CMATH"):
    try:
        dataset = load_dataset(path, split="test")
        break
    except: continue
print(f"CMATH: {len(dataset)} problems")

out_path = os.path.join(OUT_DIR, "original_results.jsonl")
summary_path = os.path.join(OUT_DIR, "original_summary.json")
fout = open(out_path, "w")
correct = total = 0
t0 = time.time()

for i, ex in enumerate(dataset):
    q = ex["question"]
    gold_raw = ex.get("golden") or ex.get("answer")
    gold = str(gold_raw).strip() if gold_raw is not None else ""

    user_msg = PROMPT.format(question=q)
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        add_generation_prompt=True, tokenize=True, return_tensors="pt",
    )

    with torch.inference_mode():
        out = model.generate(inputs=input_ids, mask_id=mask_id, **GEN_KW)

    resp = tokenizer.decode(out[0], skip_special_tokens=True)
    pred = extract_answer(resp)
    ok = normalize(pred) == normalize(gold)
    correct += ok
    total += 1

    r = dict(question=q, gold=gold, predicted=pred, correct=ok, response=resp)
    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    fout.flush()

    if total % 10 == 0 or total == len(dataset):
        elapsed = time.time() - t0
        acc = correct / total
        summary = dict(benchmark="cmath", tag="original_raw", accuracy=acc,
                       correct=correct, total=total, time_s=elapsed, done=False)
        with open(summary_path, "w") as sf:
            json.dump(summary, sf, indent=2)
        print(f"  [{total}/{len(dataset)}] acc={correct}/{total}={acc:.4f}  ({elapsed:.0f}s)")

fout.close()
elapsed = time.time() - t0
acc = correct / total
summary = dict(benchmark="cmath", tag="original_raw", accuracy=acc,
               correct=correct, total=total, time_s=elapsed, done=True)
with open(summary_path, "w") as sf:
    json.dump(summary, sf, indent=2)
print(f"\nFinal: {correct}/{total} = {acc:.4f}  ({elapsed:.0f}s)")

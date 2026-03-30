"""Prompt formatting & answer extraction helpers."""

import re
import string
import unicodedata


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def format_chat_prompt(user_message, tokenizer):
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def tokenize_prompt(text, tokenizer, device):
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    return encoded["input_ids"].to(device)


# ---------------------------------------------------------------------------
# Numeric / math answer extraction
# ---------------------------------------------------------------------------

def extract_gsm8k_answer(text):
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else ""


def normalize_numeric(s):
    s = s.strip().replace(",", "")
    try:
        return str(int(float(s)))
    except (ValueError, OverflowError):
        return s


def extract_boxed(text):
    """Extract content from \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth, start = 0, idx + 7
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None


def extract_last_number(text):
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return numbers[-1].replace(",", "").strip() if numbers else ""


def extract_math_answer(text):
    """Extract numeric answer — aligned with OpenCompass gsm8k_postprocess.

    Priority: boxed > "The answer is X" > "Answer: X" > "答案是X" > last number.
    """
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    m = re.search(r"[Tt]he answer is[:\s]*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    m = re.search(r"[Aa]nswer:\s*\$?\\?(?:boxed\{)?(-?[\d,]+(?:\.\d+)?)\}?\$?", text)
    if m:
        return m.group(1).replace(",", "").strip()
    m = re.search(r"答案[是为：:\s]+(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    return extract_last_number(text)


def normalize_math_answer(s):
    """Normalize math answer string for comparison."""
    s = s.strip()
    s = s.replace(",", "").replace(" ", "")
    s = re.sub(r"\\text\{.*?\}", "", s)
    s = re.sub(r"\\(?:mathrm|mathbf|mathit|text)\{(.*?)\}", r"\1", s)
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\%", "%").rstrip("%")
    s = s.strip()
    try:
        return str(float(s))
    except ValueError:
        return s.lower()


# ---------------------------------------------------------------------------
# Multiple choice extraction
# ---------------------------------------------------------------------------

def extract_choice_answer(text, n_choices=10):
    """Extract letter answer from model response."""
    max_letter = chr(ord("A") + n_choices - 1)
    pat = rf"[A-{max_letter}a-{max_letter.lower()}]"

    m = re.search(rf"[Tt]he answer is\s*\(?({pat})\)?", text)
    if m:
        return m.group(1).upper()
    m = re.search(rf"答案是\s*[（(]?({pat})[）)]?", text)
    if m:
        return m.group(1).upper()
    m = re.search(rf"(?:answer|选)[：:\s]*\(?({pat})\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(rf"(?:^|\n)\s*({pat})\s*$", text.strip())
    if m:
        return m.group(1).upper()
    m = re.search(rf"\b({pat})\.", text)
    if m:
        return m.group(1).upper()
    return ""


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code_block(text):
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


# ---------------------------------------------------------------------------
# QA metrics (for DROP, SQuAD, TriviaQA)
# ---------------------------------------------------------------------------

def _normalize_text(s):
    """Lower-case, remove articles / punctuation / extra whitespace."""
    s = unicodedata.normalize("NFKD", s)
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def _get_tokens(s):
    return _normalize_text(s).split()


def compute_f1(prediction, ground_truth):
    pred_tokens = _get_tokens(prediction)
    gold_tokens = _get_tokens(ground_truth)
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens) if pred_tokens else 0
    rec = len(common) / len(gold_tokens) if gold_tokens else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def compute_em(prediction, ground_truth):
    return float(_normalize_text(prediction) == _normalize_text(ground_truth))


def max_metric_over_answers(prediction, gold_answers, metric_fn):
    """Compute the max of metric_fn(pred, gold) across all gold answers."""
    if not gold_answers:
        return 0.0
    return max(metric_fn(prediction, g) for g in gold_answers)


def extract_short_answer(text):
    """Extract a short answer from the end of model output."""
    for pat in [
        r"[Tt]he answer is[:\s]+(.+?)(?:\.|$)",
        r"[Aa]nswer[：:\s]+(.+?)(?:\.|$)",
        r"####\s*(.+?)$",
    ]:
        m = re.search(pat, text.strip(), re.MULTILINE)
        if m:
            return m.group(1).strip()
    lines = text.strip().split("\n")
    return lines[-1].strip() if lines else ""

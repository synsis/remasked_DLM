"""
Offline IFEval scorer.

Usage:
    python -m eval.ifeval_scorer results_v2/ifeval/original_results.jsonl
    python -m eval.ifeval_scorer results_v2/ifeval/remask_low_prob_None_results.jsonl
"""

import json
import re
import sys


def check_instruction(instruction_id, kwargs, response):
    """Return True if response satisfies the instruction constraint."""
    kw = kwargs or {}
    resp = response or ""
    resp_lower = resp.lower()

    if instruction_id == "punctuation:no_comma":
        return "," not in resp

    elif instruction_id == "change_case:english_capital":
        return resp == resp.upper()

    elif instruction_id == "change_case:english_lowercase":
        return resp == resp.lower()

    elif instruction_id == "change_case:capital_word_frequency":
        cap_freq = kw.get("capital_frequency")
        if cap_freq is None:
            return True
        words = resp.split()
        if not words:
            return False
        cap_count = sum(1 for w in words if w[0].isupper())
        ratio = cap_count / len(words)
        capital_relation = kw.get("capital_relation", "at least")
        if capital_relation == "at least":
            return ratio >= cap_freq
        elif capital_relation == "at most":
            return ratio <= cap_freq
        return abs(ratio - cap_freq) < 0.05

    elif instruction_id == "length_constraints:number_words":
        num_words = kw.get("num_words", 0)
        relation = kw.get("relation", "at least")
        word_count = len(resp.split())
        if relation == "at least":
            return word_count >= num_words
        elif relation == "at most":
            return word_count <= num_words
        return word_count == num_words

    elif instruction_id == "length_constraints:number_sentences":
        num_sentences = kw.get("num_sentences", 0)
        relation = kw.get("relation", "at least")
        sentences = [s.strip() for s in re.split(r'[.!?]+', resp) if s.strip()]
        count = len(sentences)
        if relation == "at least":
            return count >= num_sentences
        elif relation == "at most":
            return count <= num_sentences
        return count == num_sentences

    elif instruction_id == "length_constraints:number_paragraphs":
        num_paragraphs = kw.get("num_paragraphs", 0)
        paragraphs = [p.strip() for p in resp.split("\n\n") if p.strip()]
        return len(paragraphs) >= num_paragraphs

    elif instruction_id == "length_constraints:nth_paragraph_first_word":
        nth = kw.get("nth_paragraph", 1)
        first_word = kw.get("first_word", "")
        paragraphs = [p.strip() for p in resp.split("\n\n") if p.strip()]
        if nth > len(paragraphs):
            return False
        para = paragraphs[nth - 1]
        words = para.split()
        if not words:
            return False
        return words[0].lower() == first_word.lower()

    elif instruction_id == "detectable_format:number_bullet_lists":
        num_bullets = kw.get("num_bullets", 0)
        bullets = re.findall(r'(?m)^\s*[\*\-\•]\s+', resp)
        numbered = re.findall(r'(?m)^\s*\d+[\.\)]\s+', resp)
        return len(bullets) + len(numbered) >= num_bullets

    elif instruction_id == "detectable_format:number_highlighted_sections":
        num_highlights = kw.get("num_highlights", 0)
        highlights = re.findall(r'\*[^*]+\*', resp)
        return len(highlights) >= num_highlights

    elif instruction_id == "detectable_format:multiple_sections":
        section_spliter = kw.get("section_spliter", "Section")
        num_sections = kw.get("num_sections", 0)
        pattern = re.escape(section_spliter)
        sections = re.findall(pattern, resp, re.IGNORECASE)
        return len(sections) >= num_sections

    elif instruction_id == "detectable_format:json_format":
        try:
            json.loads(resp)
            return True
        except (json.JSONDecodeError, ValueError):
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', resp, re.DOTALL)
            if json_match:
                try:
                    json.loads(json_match.group(1))
                    return True
                except (json.JSONDecodeError, ValueError):
                    pass
            return False

    elif instruction_id == "detectable_format:title":
        return bool(re.search(r'(?m)^#\s+\S|^[A-Z][^\n]{2,}$', resp))

    elif instruction_id == "detectable_format:constrained_response":
        return len(resp.split()) <= 5 or resp.strip() in ("Yes", "No", "True", "False")

    elif instruction_id == "detectable_content:postscript":
        postscript_marker = kw.get("postscript_marker", "P.S.")
        return postscript_marker in resp

    elif instruction_id == "detectable_content:number_placeholders":
        num_placeholders = kw.get("num_placeholders", 0)
        placeholders = re.findall(r'\[.*?\]', resp)
        return len(placeholders) >= num_placeholders

    elif instruction_id == "keywords:existence":
        keywords = kw.get("keywords", [])
        return all(k.lower() in resp_lower for k in keywords)

    elif instruction_id == "keywords:forbidden_words":
        forbidden = kw.get("forbidden_words", [])
        return all(w.lower() not in resp_lower for w in forbidden)

    elif instruction_id == "keywords:frequency":
        keyword = kw.get("keyword", "")
        frequency = kw.get("frequency", 0)
        relation = kw.get("relation", "at least")
        count = resp_lower.count(keyword.lower())
        if relation == "at least":
            return count >= frequency
        elif relation == "at most":
            return count <= frequency
        return count == frequency

    elif instruction_id == "keywords:letter_frequency":
        letter = kw.get("letter", "")
        let_frequency = kw.get("let_frequency", 0)
        relation = kw.get("let_relation", "at least")
        count = resp_lower.count(letter.lower())
        if relation == "at least":
            return count >= let_frequency
        elif relation == "at most":
            return count <= let_frequency
        return count == let_frequency

    elif instruction_id == "language:response_language":
        language = kw.get("language", "").lower()
        if language == "english":
            ascii_ratio = sum(1 for c in resp if ord(c) < 128) / max(len(resp), 1)
            return ascii_ratio > 0.8
        return True

    elif instruction_id == "combination:repeat_prompt":
        return True

    elif instruction_id == "combination:two_responses":
        separators = ["******", "---", "===", "***"]
        return any(sep in resp for sep in separators)

    elif instruction_id == "startend:end_checker":
        end_phrase = kw.get("end_phrase", "")
        return resp.rstrip().endswith(end_phrase)

    elif instruction_id == "startend:quotation":
        stripped = resp.strip()
        return (stripped.startswith('"') and stripped.endswith('"')) or \
               (stripped.startswith("'") and stripped.endswith("'"))

    return True


def score_file(path):
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    prompt_strict = 0
    prompt_loose = 0
    inst_strict = 0
    inst_loose = 0
    total_prompts = len(results)
    total_insts = 0

    for r in results:
        ids = r.get("instruction_id_list", [])
        kwargs_list = r.get("kwargs", [])
        resp = r.get("response", "")

        all_pass = True
        any_fail = False
        for j, iid in enumerate(ids):
            kw = kwargs_list[j] if j < len(kwargs_list) else {}
            ok = check_instruction(iid, kw, resp)
            total_insts += 1
            if ok:
                inst_strict += 1
                inst_loose += 1
            else:
                any_fail = True
                inst_loose += 0
                all_pass = False

        if all_pass:
            prompt_strict += 1
        if not any_fail:
            prompt_loose += 1

    prompt_acc = prompt_strict / total_prompts if total_prompts else 0
    inst_acc = inst_strict / total_insts if total_insts else 0

    print(f"File: {path}")
    print(f"  Prompt-level (strict): {prompt_strict}/{total_prompts} = {prompt_acc*100:.2f}%")
    print(f"  Instruction-level:     {inst_strict}/{total_insts} = {inst_acc*100:.2f}%")
    return prompt_acc, inst_acc, prompt_strict, total_prompts, inst_strict, total_insts


if __name__ == "__main__":
    for path in sys.argv[1:]:
        score_file(path)
        print()

"""
Capture clean trajectory for paper figure.
max_remask_per_pos=2 for readable visualization.
"""
import json, time, gc, os, sys, torch, torch.nn.functional as F
import remask.env
from remask.loader import load_original_model, load_remask_model
from remask.utils import format_chat_prompt, tokenize_prompt, extract_math_answer

MASK_ID, EOS_ID = 156895, 156892

AIME_PROMPT = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $ANSWER (without quotes) where "
    "$ANSWER is the answer to the problem.\n\n{problem}\n\n"
    "Remember to put your answer on its own line after \"Answer:\"."
)

DROP_PROMPT = (
    "Read the passage and answer the question. "
    "Give only the final answer as briefly as possible (a number, name, or short phrase). "
    'Use the format "The answer is <answer>".\n\n'
    "Passage: {passage}\n\nQuestion: {question}\n\nAnswer:"
)


@torch.inference_mode()
def gen_with_traj(model, input_ids, tokenizer, block_length=32, gen_length=512,
                  threshold=0.7, editing_threshold=0.5):
    device = model.device
    input_ids = input_ids.to(device)
    prompt_length = input_ids.shape[1]
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    position_ids = torch.arange(total_length, device=device).unsqueeze(0)
    x = torch.full((1, total_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids.clone()

    prefill_blocks = prompt_length // block_length
    kv_cache = None
    trajectory = []

    if hasattr(model, '_prev_logits'):
        model._prev_logits = None
        model._remask_counts = None

    for num_block in range(prefill_blocks, num_blocks):
        current_window_end = (num_block + 1) * block_length
        block_start = num_block * block_length
        prefix_len = block_start

        prompt_mask_in_block = torch.zeros(block_length, dtype=torch.bool, device=device)
        if block_start < prompt_length:
            prompt_mask_in_block[:min(prompt_length - block_start, block_length)] = True

        block_traj = {"block_idx": num_block - prefill_blocks, "iterations": []}

        if hasattr(model, '_prev_logits'):
            model._prev_logits = None
            model._remask_counts = None

        post_steps = 0
        cache_valid = (kv_cache is not None
                       and kv_cache.get_seq_length() == prefix_len
                       and prefix_len > 0)
        iteration = 0

        while True:
            old_block_tokens = x[:, block_start:current_window_end].clone()
            active_block_mask = old_block_tokens == MASK_ID
            if not active_block_mask.any():
                post_steps += 1
            if post_steps > 16:
                break

            if not cache_valid:
                cur_x = x[:, :current_window_end]
                nb = (current_window_end + block_length - 1) // block_length
                bm = torch.tril(torch.ones(nb, nb, device=device))
                cur_attn = (bm.repeat_interleave(block_length, 0)
                              .repeat_interleave(block_length, 1)
                            [:current_window_end, :current_window_end]
                            .unsqueeze(0).unsqueeze(0).to(torch.bfloat16))
                cur_pos = position_ids[:, :current_window_end]
                active_logits, kv_cache = model._forward_block(
                    cur_x, cur_pos, cur_attn, None, block_length)
                cache_valid = True
            else:
                model._truncate_kv_cache(kv_cache, prefix_len)
                block_ids = x[:, block_start:current_window_end]
                block_pos = position_ids[:, block_start:current_window_end]
                cache_attn = torch.ones(1, 1, block_length, prefix_len + block_length,
                                        dtype=torch.bfloat16, device=device)
                active_logits, kv_cache = model._forward_block(
                    block_ids, block_pos, cache_attn, kv_cache, block_length)

            x0, x0_p = model._sample_with_temperature_topk_topp(
                active_logits, temperature=0.0, top_k=None, top_p=None)

            probs = F.softmax(active_logits[0].float(), dim=-1)
            tokens_before = old_block_tokens[0].cpu().tolist()
            pos_probs, top1_tokens, top1_probs = [], [], []
            for pos_i in range(block_length):
                tok = tokens_before[pos_i]
                top1_p, top1_id = probs[pos_i].max(dim=-1)
                top1_tokens.append(top1_id.item())
                top1_probs.append(round(top1_p.item(), 4))
                pos_probs.append(0.0 if tok == MASK_ID else round(probs[pos_i, tok].item(), 4))

            mask_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            if active_block_mask.sum() > 0:
                mask_confidence = torch.where(
                    active_block_mask, x0_p,
                    torch.tensor(-torch.inf, device=device))
                high_conf_mask = (mask_confidence[0] > threshold) & active_block_mask[0]
                if high_conf_mask.sum().item() >= 1:
                    mask_transfer_index[0] = high_conf_mask
                else:
                    n_avail = active_block_mask.sum().item()
                    if n_avail > 0:
                        _, idx = torch.topk(mask_confidence[0], k=min(1, n_avail))
                        mask_transfer_index[0, idx] = True

            block_view = x[:, block_start:current_window_end]
            should_break = model._token_edit_step(
                block_view, x0, x0_p, active_logits, mask_transfer_index,
                old_block_tokens, active_block_mask,
                prompt_mask_in_block, editing_threshold,
                block_length, MASK_ID)

            tokens_after = x[0, block_start:current_window_end].cpu().tolist()
            states = []
            for pos_i in range(block_length):
                old_tok = tokens_before[pos_i]
                new_tok = tokens_after[pos_i]
                if prompt_mask_in_block[pos_i]:
                    states.append("prompt")
                elif new_tok == MASK_ID and old_tok == MASK_ID:
                    states.append("mask")
                elif new_tok == MASK_ID and old_tok != MASK_ID:
                    states.append("remasked")
                elif old_tok == MASK_ID and new_tok != MASK_ID:
                    states.append("filled")
                else:
                    states.append("committed")

            block_traj["iterations"].append({
                "iter": iteration,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "states": states,
                "pos_probs": pos_probs,
                "top1_tokens": top1_tokens,
                "top1_probs": top1_probs,
                "n_mask": sum(1 for s in states if s == "mask"),
                "n_remasked": sum(1 for s in states if s == "remasked"),
                "n_filled": sum(1 for s in states if s == "filled"),
            })
            iteration += 1
            if active_block_mask.sum() == 0 and should_break:
                break
            if iteration > 200:
                break

        trajectory.append(block_traj)
        gen_part = x[0, prompt_length:current_window_end]
        if (gen_part != MASK_ID).all() and (gen_part == EOS_ID).any():
            break

    output_ids = x[0, prompt_length:].cpu().tolist()
    eos_pos = next((i for i, t in enumerate(output_ids) if t == EOS_ID), None)
    if eos_pos is not None:
        output_ids = output_ids[:eos_pos]
    output_ids = [t for t in output_ids if t != MASK_ID]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    return response, trajectory


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "drop"
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    gen_length = int(sys.argv[3]) if len(sys.argv) > 3 else 512

    if dataset == "drop":
        from datasets import load_dataset
        ds = load_dataset("drop", split="validation")
        ex = ds[idx]
        gold_spans = ex.get("answers_spans", {}).get("spans", [])
        gold = gold_spans[0] if gold_spans else "?"
        prompt_tpl = DROP_PROMPT.format(passage=ex["passage"], question=ex["question"])
        question = ex["question"]
    elif dataset == "aime":
        with open("data/aime2025.json") as f:
            aime = json.load(f)
        ex = aime[idx]
        gold = str(ex["answer"])
        prompt_tpl = AIME_PROMPT.format(problem=ex["problem"])
        question = ex["problem"][:100]

    print(f"[{dataset}:{idx}] Q: {question[:80]}...")
    print(f"Gold: {gold}", flush=True)

    # ---- T2T ----
    print("\n=== Loading T2T model ===", flush=True)
    model, tokenizer, _ = load_original_model()
    prompt_text = format_chat_prompt(prompt_tpl, tokenizer)
    input_ids = tokenize_prompt(prompt_text, tokenizer, model.device)
    print(f"Prompt: {input_ids.shape[1]} tokens", flush=True)

    print("Running T2T...", flush=True)
    t0 = time.time()
    resp_t2t, traj_t2t = gen_with_traj(model, input_ids, tokenizer, gen_length=gen_length)
    print(f"T2T done {time.time()-t0:.1f}s: {repr(resp_t2t[:80])}", flush=True)

    del model; gc.collect()
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i): torch.cuda.empty_cache()

    # ---- T2M (max_remask_per_pos=2) ----
    print("\n=== Loading T2M model (max_remask=2) ===", flush=True)
    model, tokenizer, _ = load_remask_model(
        strategy="low_prob", remask_threshold=None,
        max_remask_per_pos=2, max_remask_ratio=0.25)
    input_ids = tokenize_prompt(prompt_text, tokenizer, model.device)

    print("Running T2M...", flush=True)
    t0 = time.time()
    resp_t2m, traj_t2m = gen_with_traj(model, input_ids, tokenizer, gen_length=gen_length)
    print(f"T2M done {time.time()-t0:.1f}s: {repr(resp_t2m[:80])}", flush=True)

    del model; gc.collect()
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i): torch.cuda.empty_cache()

    # Summary
    print(f"\nT2T: {repr(resp_t2t[:100])}")
    print(f"T2M: {repr(resp_t2m[:100])}")
    diff = "SAME" if resp_t2t.strip() == resp_t2m.strip() else "DIFFERENT"
    print(f"Result: {diff}")

    for mode, traj in [("T2T", traj_t2t), ("T2M", traj_t2m)]:
        total_rm = sum(sum(it["n_remasked"] for it in blk["iterations"]) for blk in traj)
        print(f"  {mode}: {len(traj)} blocks, total_remask={total_rm}")

    out_path = f"results_v2/trajectory_final_{dataset}_{idx}.json"
    data = {
        "dataset": dataset, "idx": idx,
        "question": question, "gold": gold,
        "t2t_response": resp_t2t, "t2m_response": resp_t2m,
        "t2t_trajectory": traj_t2t, "t2m_trajectory": traj_t2m,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

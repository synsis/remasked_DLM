#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  通用多卡 shard 并行 eval 脚本
#  用法:
#    bash run_eval_8gpu.sh <dataset> [bsz] [ngpu]
#
#  例子:
#    bash run_eval_8gpu.sh hellaswag           # 默认 bsz=16, 全部GPU
#    bash run_eval_8gpu.sh drop 32             # bsz=32, 全部GPU
#    bash run_eval_8gpu.sh triviaqa 16 4       # bsz=16, 只用4卡
#    bash run_eval_8gpu.sh gsm_plus 16 8       # bsz=16, 8卡
#    bash run_eval_8gpu.sh cmath 64 8
#    bash run_eval_8gpu.sh bbh 16 8
#    bash run_eval_8gpu.sh piqa 64 4
#    bash run_eval_8gpu.sh aime2025 4 2
#
#  支持的 dataset: cmath, gsm_plus, piqa, hellaswag, triviaqa, drop, bbh, aime2025, humaneval, mbpp
#  跑完后 results_v2/<dataset>/ 下会有:
#    original_results.jsonl          汇总结果
#    original_summary.json           汇总 summary
#    remask_low_prob_None_results.jsonl
#    remask_low_prob_None_summary.json
# ═══════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")"

export https_proxy="100.68.162.212:3128"
export http_proxy="100.68.162.212:3128"

DATASET="${1:?用法: bash run_eval_8gpu.sh <dataset> [bsz] [ngpu]}"
BSZ="${2:-16}"
NGPU="${3:-0}"  # 0 = 自动检测全部
ENV="${CONDA_ENV:-remask}"
OUT="results_v2/${DATASET}"

# 检测可用 GPU
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra ALL_GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  ALL_GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' '))
fi

# 如果指定了 ngpu, 截取前 N 张
if [ "$NGPU" -gt 0 ] && [ "$NGPU" -lt "${#ALL_GPUS[@]}" ]; then
  ALL_GPUS=("${ALL_GPUS[@]:0:$NGPU}")
fi
SHARDS=${#ALL_GPUS[@]}
echo "Dataset: $DATASET  BSZ: $BSZ  GPUs: $SHARDS (${ALL_GPUS[*]})"

mkdir -p "$OUT"

run_mode() {
  local mode=$1
  local extra_args="${2:-}"
  echo ""
  echo "========== $DATASET  mode=$mode  bsz=$BSZ  shards=$SHARDS =========="

  pids=()
  for ((i=0; i<SHARDS; i++)); do
    gpu_id=${ALL_GPUS[$i]}
    echo "[GPU $gpu_id] shard $i/$SHARDS  mode=$mode"
    CUDA_VISIBLE_DEVICES=$gpu_id conda run -n $ENV \
      python -m eval.${DATASET} \
        --output_dir "$OUT" \
        --mode "$mode" \
        --batch_size $BSZ \
        --shard_id $i \
        --num_shards $SHARDS \
        $extra_args \
      > "$OUT/${mode}_shard${i}.log" 2>&1 &
    pids+=($!)
  done

  failed=0
  for ((i=0; i<SHARDS; i++)); do
    wait "${pids[$i]}" 2>/dev/null || true
    ec=$?
    gpu_id=${ALL_GPUS[$i]}
    if [ $ec -ne 0 ]; then
      echo "[GPU $gpu_id] shard $i FAILED exit=$ec  (see $OUT/${mode}_shard${i}.log)"
      failed=$((failed + 1))
    else
      echo "[GPU $gpu_id] shard $i done"
    fi
  done
  echo "mode=$mode: $((SHARDS - failed))/$SHARDS shards succeeded"
  [ $failed -gt 0 ] && echo "WARNING: $failed shards failed. 降低 bsz 重跑: bash $0 $DATASET $((BSZ/2))"
}

merge() {
  python3 << 'PYEOF'
import json, os, sys

dataset = os.environ["DATASET"]
out_dir = os.environ["OUT"]
shards = int(os.environ["SHARDS"])
benchmark = dataset

for tag, mode in [("original", "original"), ("remask_low_prob_None", "remask")]:
    all_results = []
    missing = []
    for s in range(shards):
        p = os.path.join(out_dir, f"{tag}_shard{s}_results.jsonl")
        if not os.path.exists(p):
            missing.append(s)
            continue
        with open(p) as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

    if not all_results:
        print(f"  {tag}: no results")
        continue
    if missing:
        print(f"  WARNING {tag}: shards {missing} missing!")

    merged_jsonl = os.path.join(out_dir, f"{tag}_results.jsonl")
    with open(merged_jsonl, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    correct = sum(1 for r in all_results if r.get("correct"))
    total = len(all_results)
    acc = correct / total if total else 0

    # F1/EM for drop/triviaqa
    has_f1 = any("f1" in r for r in all_results)
    has_em = any("em" in r for r in all_results)

    params = {}
    for s in range(shards):
        sp = os.path.join(out_dir, f"{tag}_shard{s}_summary.json")
        if os.path.exists(sp):
            params = json.load(open(sp))
            break

    summary = dict(
        benchmark=benchmark, tag=tag, mode=mode,
        accuracy=acc, correct=correct, total=total,
        done=True, shards=shards,
    )
    if has_f1:
        summary["avg_f1"] = sum(r.get("f1", 0) for r in all_results) / total
    if has_em:
        summary["avg_em"] = sum(r.get("em", 0) for r in all_results) / total

    for k in ("gen_length", "block_length", "steps", "threshold", "editing_threshold",
              "temperature", "batch_size", "strategy", "remask_threshold"):
        if k in params:
            summary[k] = params[k]

    shard_times = []
    for s in range(shards):
        sp = os.path.join(out_dir, f"{tag}_shard{s}_summary.json")
        if os.path.exists(sp):
            shard_times.append(json.load(open(sp)).get("time_s", 0))
    if shard_times:
        summary["time_s_max_shard"] = max(shard_times)
        summary["time_s_sum_shards"] = sum(shard_times)

    tpfs = [r["_tpf"] for r in all_results if r.get("_tpf") is not None]
    tpss = [r["_tps"] for r in all_results if r.get("_tps") is not None]
    fwds = [r["_forward_passes"] for r in all_results if r.get("_forward_passes") is not None]
    toks = [r["_output_tokens"] for r in all_results if r.get("_output_tokens") is not None]
    remasks = [r["_remask_total"] for r in all_results if r.get("_remask_total") is not None]
    if tpfs: summary["avg_tpf"] = sum(tpfs) / len(tpfs)
    if tpss: summary["avg_tps"] = sum(tpss) / len(tpss)
    if fwds:
        summary["avg_forward_passes"] = sum(fwds) / len(fwds)
        summary["total_forward_passes"] = sum(fwds)
    if toks:
        summary["avg_output_tokens"] = sum(toks) / len(toks)
        summary["total_output_tokens"] = sum(toks)
    if remasks:
        summary["avg_remask_total"] = sum(remasks) / len(remasks)
        summary["total_remask"] = sum(remasks)

    merged_summary = os.path.join(out_dir, f"{tag}_summary.json")
    with open(merged_summary, "w") as f:
        json.dump(summary, f, indent=2)

    metric = f"acc={acc*100:.2f}%"
    if has_f1:
        metric += f"  F1={summary['avg_f1']*100:.2f}%"
    if has_em:
        metric += f"  EM={summary['avg_em']*100:.2f}%"
    print(f"  {tag}: {correct}/{total}  {metric}  (wall={summary.get('time_s_max_shard',0):.0f}s)")

    # Merge _samples.jsonl for coding tasks (humaneval/mbpp)
    sample_files = [os.path.join(out_dir, f"{tag}_shard{s}_samples.jsonl") for s in range(shards)]
    sample_files = [p for p in sample_files if os.path.exists(p)]
    if sample_files:
        all_samples = []
        for p in sample_files:
            with open(p) as f:
                for line in f:
                    if line.strip():
                        all_samples.append(json.loads(line))
        merged_samples = os.path.join(out_dir, f"{tag}_samples.jsonl")
        with open(merged_samples, "w") as f:
            for s in all_samples:
                f.write(json.dumps(s) + "\n")
        print(f"  {tag}: {len(all_samples)} samples merged → {merged_samples}")
PYEOF
}

run_evalplus() {
  # Run evalplus for coding benchmarks (humaneval/mbpp)
  case "$DATASET" in
    humaneval|mbpp) ;;
    *) return ;;
  esac
  echo ""
  echo "===== Running evalplus ($DATASET) ====="
  for tag in original remask_low_prob_None; do
    local samples="$OUT/${tag}_samples.jsonl"
    if [ ! -f "$samples" ]; then
      echo "  $tag: no samples, skip"
      continue
    fi
    echo "  evalplus: $tag ..."
    conda run -n $ENV evalplus.evaluate \
      --dataset "$DATASET" \
      --samples "$samples" \
      2>&1 | tee "$OUT/${tag}_evalplus.log"
  done
}

run_mode "original"
run_mode "remask" "--strategy low_prob"

echo ""
echo "===== Merging shards ====="
export DATASET OUT SHARDS
merge

run_evalplus

echo ""
echo "===== Done: $DATASET ====="
echo "Results: $OUT/original_summary.json"
echo "         $OUT/remask_low_prob_None_summary.json"

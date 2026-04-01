#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  通用多卡 shard 并行 eval 脚本 (v2: 支持选择 mode, OOM 自动重试)
#  用法:
#    bash run_eval_8gpu2.sh <dataset> [bsz] [ngpu] [original|remask|all] [gen_length]
#
#  例子:
#    bash run_eval_8gpu2.sh hellaswag                       # bsz=16, 全部GPU, all
#    bash run_eval_8gpu2.sh drop 32 8 all                   # bsz=32, 8卡, 两个都跑
#    bash run_eval_8gpu2.sh bbh 16 8 remask                 # bsz=16, 8卡, 只跑remask
#    bash run_eval_8gpu2.sh mmlu_pro 16 4 original          # bsz=16, 4卡, 只跑original
#    bash run_eval_8gpu2.sh humaneval 1 4 remask            # 代码任务, 只跑remask
#    bash run_eval_8gpu2.sh bbh 128 8 remask 8192           # 指定 gen_length=8192
#
#  OOM 自动重试: shard OOM 后自动减半 bsz 重跑, 直到成功或 bsz<1
#
#  支持的 dataset: cmath, gsm_plus, piqa, hellaswag, triviaqa, drop, bbh,
#    aime2025, humaneval, mbpp, mmlu_pro, gpqa, ifeval, bfcl
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
export HF_TOKEN="${HF_TOKEN:-}"

DATASET="${1:?用法: bash run_eval_8gpu2.sh <dataset> [bsz] [ngpu] [original|remask|all] [gen_length]}"
BSZ="${2:-16}"
NGPU="${3:-0}"  # 0 = 自动检测全部
RUN_MODE="${4:-all}"  # original / remask / all
GEN_LENGTH="${5:-}"   # 空 = 用 eval 脚本默认值 (16384)
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

GEN_LEN_ARGS=""
[ -n "$GEN_LENGTH" ] && GEN_LEN_ARGS="--gen_length $GEN_LENGTH"

echo "Dataset: $DATASET  BSZ: $BSZ  GPUs: $SHARDS (${ALL_GPUS[*]})  Mode: $RUN_MODE  gen_length: ${GEN_LENGTH:-default}"

mkdir -p "$OUT"

is_oom() {
  grep -q "OutOfMemoryError\|CUDA out of memory" "$1" 2>/dev/null
}

launch_shard() {
  local gpu_id=$1 shard_id=$2 bsz=$3 mode=$4 extra_args="$5" logfile="$6"
  CUDA_VISIBLE_DEVICES=$gpu_id PYTHONUNBUFFERED=1 conda run -n $ENV \
    python -u -m eval.${DATASET} \
      --output_dir "$OUT" \
      --mode "$mode" \
      --batch_size $bsz \
      --shard_id $shard_id \
      --num_shards $SHARDS \
      $GEN_LEN_ARGS \
      $extra_args \
    > "$logfile" 2>&1
}

run_mode() {
  local mode=$1
  local extra_args="${2:-}"
  echo ""
  echo "========== $DATASET  mode=$mode  bsz=$BSZ  shards=$SHARDS =========="

  # --- 第一轮: 全部 shard 并行 ---
  pids=()
  for ((i=0; i<SHARDS; i++)); do
    gpu_id=${ALL_GPUS[$i]}
    echo "[GPU $gpu_id] shard $i/$SHARDS  mode=$mode  bsz=$BSZ"
    launch_shard "$gpu_id" "$i" "$BSZ" "$mode" "$extra_args" "$OUT/${mode}_shard${i}.log" &
    pids+=($!)
  done

  # 收集结果, 记录 OOM 的 shard
  oom_shards=()
  ok=0
  non_oom_fail=0
  for ((i=0; i<SHARDS; i++)); do
    wait "${pids[$i]}" 2>/dev/null || true
    ec=$?
    gpu_id=${ALL_GPUS[$i]}
    if [ $ec -eq 0 ]; then
      echo "[GPU $gpu_id] shard $i done"
      ok=$((ok + 1))
    elif is_oom "$OUT/${mode}_shard${i}.log"; then
      echo "[GPU $gpu_id] shard $i OOM (bsz=$BSZ)"
      oom_shards+=($i)
    else
      echo "[GPU $gpu_id] shard $i FAILED exit=$ec (non-OOM, see $OUT/${mode}_shard${i}.log)"
      non_oom_fail=$((non_oom_fail + 1))
    fi
  done

  # --- OOM 重试: 每轮减半 bsz, OOM 的 shard 并行重跑 ---
  local retry_bsz=$BSZ
  while [ ${#oom_shards[@]} -gt 0 ]; do
    retry_bsz=$((retry_bsz / 2))
    if [ $retry_bsz -lt 1 ]; then
      echo "ERROR: bsz reduced to 0, giving up on shards: ${oom_shards[*]}"
      non_oom_fail=$((non_oom_fail + ${#oom_shards[@]}))
      break
    fi
    echo ""
    echo "--- OOM retry: shards [${oom_shards[*]}]  bsz=$retry_bsz ---"

    retry_pids=()
    retry_indices=("${oom_shards[@]}")
    oom_shards=()
    for i in "${retry_indices[@]}"; do
      gpu_id=${ALL_GPUS[$i]}
      echo "[GPU $gpu_id] shard $i retrying  bsz=$retry_bsz"
      launch_shard "$gpu_id" "$i" "$retry_bsz" "$mode" "$extra_args" "$OUT/${mode}_shard${i}.log" &
      retry_pids+=($!)
    done

    for idx in "${!retry_indices[@]}"; do
      i=${retry_indices[$idx]}
      wait "${retry_pids[$idx]}" 2>/dev/null || true
      ec=$?
      gpu_id=${ALL_GPUS[$i]}
      if [ $ec -eq 0 ]; then
        echo "[GPU $gpu_id] shard $i done (bsz=$retry_bsz)"
        ok=$((ok + 1))
      elif is_oom "$OUT/${mode}_shard${i}.log"; then
        echo "[GPU $gpu_id] shard $i OOM again (bsz=$retry_bsz)"
        oom_shards+=($i)
      else
        echo "[GPU $gpu_id] shard $i FAILED exit=$ec (non-OOM)"
        non_oom_fail=$((non_oom_fail + 1))
      fi
    done
  done

  local total_fail=$((non_oom_fail))
  echo "mode=$mode: $ok/$SHARDS shards succeeded"
  [ $total_fail -gt 0 ] && echo "WARNING: $total_fail shards failed (non-OOM or bsz exhausted)"
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

if [ "$RUN_MODE" = "original" ] || [ "$RUN_MODE" = "all" ]; then
  run_mode "original"
fi

if [ "$RUN_MODE" = "remask" ] || [ "$RUN_MODE" = "all" ]; then
  run_mode "remask" "--strategy low_prob"
fi

echo ""
echo "===== Merging shards ====="
export DATASET OUT SHARDS
merge

run_evalplus

echo ""
echo "===== Done: $DATASET ====="
echo "Results: $OUT/original_summary.json"
echo "         $OUT/remask_low_prob_None_summary.json"

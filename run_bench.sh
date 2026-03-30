#!/usr/bin/env bash
# ── Benchmark: original-only (align with paper) ──
# gen_length=16384 (default), eos_early_stop handles early exit.
# Q-Mode: threshold=0.7, editing_threshold=0.5
set -euo pipefail
cd "$(dirname "$0")"

export https_proxy="100.68.162.212:3128"
export http_proxy="100.68.162.212:3128"

CONDA_ENV="${CONDA_ENV:-remask}"
OUT="results"

TASKS=()
add() { local d="$OUT/$1"; mkdir -p "$d"; shift; TASKS+=("$@"); }

# 1. AIME 2025 (30 problems) — fast
add aime2025 \
  "python -m eval.aime2025 --output_dir $OUT/aime2025 --mode original"

# 2. CMATH (~1098) — Chinese math
add cmath \
  "python -m eval.cmath --output_dir $OUT/cmath --mode original"

# 3. PIQA (~1838) — physical intuition
add piqa \
  "python -m eval.piqa --output_dir $OUT/piqa --mode original"

# 4. GSM-Plus testmini (~2400) — math
add gsm_plus \
  "python -m eval.gsm_plus --output_dir $OUT/gsm_plus --mode original"

# 5. HellaSwag (~10K) — reasoning
add hellaswag \
  "python -m eval.hellaswag --output_dir $OUT/hellaswag --mode original"

# 6. TriviaQA — knowledge
add triviaqa \
  "python -m eval.triviaqa --output_dir $OUT/triviaqa --mode original"

# 7. DROP — reading comprehension
add drop \
  "python -m eval.drop --output_dir $OUT/drop --mode original"

# 8. BBH — reasoning
add bbh \
  "python -m eval.bbh --output_dir $OUT/bbh --mode original"

# ═══════════════════════════════════════════════════
#  GPU SCHEDULER
# ═══════════════════════════════════════════════════
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' '))
fi
NUM_GPUS=${#GPUS[@]}

echo "Tasks: ${#TASKS[@]}  |  GPUs: ${NUM_GPUS} (${GPUS[*]})"
echo ""

declare -A gpu_pid
declare -A gpu_task_desc
NEXT_TASK=0
DONE=0
TOTAL=${#TASKS[@]}
FAILED=0

assign_task() {
  local gpu_idx=$1
  local gpu_id=${GPUS[$gpu_idx]}
  if [ $NEXT_TASK -ge $TOTAL ]; then
    return 1
  fi
  local task="${TASKS[$NEXT_TASK]}"
  local task_id=$NEXT_TASK
  NEXT_TASK=$((NEXT_TASK + 1))
  echo "[GPU $gpu_id] Starting task $((task_id+1))/$TOTAL: $task"
  CUDA_VISIBLE_DEVICES=$gpu_id conda run -n $CONDA_ENV $task &
  gpu_pid[$gpu_idx]=$!
  gpu_task_desc[$gpu_idx]="$task"
  return 0
}

for ((i=0; i<NUM_GPUS && i<TOTAL; i++)); do
  assign_task $i
done

while [ $DONE -lt $TOTAL ]; do
  for ((i=0; i<NUM_GPUS; i++)); do
    pid=${gpu_pid[$i]:-0}
    [ "$pid" -eq 0 ] && continue
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null
      exit_code=$?
      DONE=$((DONE + 1))
      gpu_id=${GPUS[$i]}
      if [ $exit_code -eq 0 ]; then
        echo "[GPU $gpu_id] Task done ($DONE/$TOTAL)"
      else
        echo "[GPU $gpu_id] Task FAILED exit=$exit_code ($DONE/$TOTAL): ${gpu_task_desc[$i]}"
        FAILED=$((FAILED + 1))
      fi
      gpu_pid[$i]=0
      assign_task $i || true
    fi
  done
  sleep 2
done

echo ""
echo "All $TOTAL tasks complete ($FAILED failed)"
echo ""

# ── Paper reference (LLaDA2.1-mini Q-Mode) ──
echo "Paper reference:  AIME=43.33  CMATH=94.99  GSM-Plus=86.55  PIQA=86.89  HellaSwag=76.19  TriviaQA=54.24  DROP=82.37  BBH=80.58"
echo ""

python3 -c "
import json, glob
results = []
for f in sorted(glob.glob('$OUT/*/*_summary.json')):
    try:
        d = json.load(open(f))
        results.append(d)
    except Exception:
        pass
results.sort(key=lambda x: (x.get('benchmark',''), x.get('tag','')))
print(f\"{'Benchmark':<14} {'Config':<35} {'Acc':>8} {'N':>6} {'Time':>8}\")
print('='*75)
for d in results:
    bm = d.get('benchmark','?')
    tag = d.get('tag','?')
    n = d.get('total',0)
    t = d.get('time_s',0)
    acc_val = d.get('accuracy')
    if acc_val is not None:
        metric = f'{acc_val*100:.1f}%'
    elif d.get('correct') is not None:
        metric = f\"{d['correct']}/{n}\"
    else:
        metric = '—'
    print(f\"{bm:<14} {tag:<35} {metric:>8} {n:>6} {t:>7.0f}s\")
"

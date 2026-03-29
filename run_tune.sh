#!/usr/bin/env bash
# ── Dynamic GPU Job Scheduler ──
# Assigns tasks to GPUs as they become free.
set -euo pipefail
cd "$(dirname "$0")"

CONDA_ENV="${CONDA_ENV:-remask}"
N=50
OUTDIR="results_tune/gsm_plus"
mkdir -p "$OUTDIR"

BASE="python -m eval.gsm_plus --max_samples $N --output_dir $OUTDIR"

# ═══════════════════════════════════════════════════
#  TASK LIST
# ═══════════════════════════════════════════════════
TASKS=(
  # Baseline: original LLaDA2.1-mini (no remask)
  "$BASE --mode original"
  # Strategy 1: low_prob — threshold ∈ {0.3, 0.5, 0.7, 0.9}
  "$BASE --mode remask --strategy low_prob --remask_threshold 0.3"
  "$BASE --mode remask --strategy low_prob --remask_threshold 0.5"
  "$BASE --mode remask --strategy low_prob --remask_threshold 0.7"
  "$BASE --mode remask --strategy low_prob --remask_threshold 0.9"
  # Strategy 2: t2t_remask — threshold ∈ {0.7, 0.8, 0.9, 0.95}
  "$BASE --mode remask --strategy t2t_remask --remask_threshold 0.7"
  "$BASE --mode remask --strategy t2t_remask --remask_threshold 0.8"
  "$BASE --mode remask --strategy t2t_remask --remask_threshold 0.9"
  "$BASE --mode remask --strategy t2t_remask --remask_threshold 0.95"
  # Strategy 3: logit_diff — threshold ∈ {0.1, 0.2, 0.3, 0.5}
  "$BASE --mode remask --strategy logit_diff --remask_threshold 0.1"
  "$BASE --mode remask --strategy logit_diff --remask_threshold 0.2"
  "$BASE --mode remask --strategy logit_diff --remask_threshold 0.3"
  "$BASE --mode remask --strategy logit_diff --remask_threshold 0.5"
)

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
  echo "[GPU $gpu_id] Starting task $((task_id+1))/$TOTAL"
  CUDA_VISIBLE_DEVICES=$gpu_id conda run -n $CONDA_ENV $task &
  gpu_pid[$gpu_idx]=$!
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
        echo "[GPU $gpu_id] Task FAILED exit=$exit_code ($DONE/$TOTAL)"
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

# Print all results (including previous strategy 1 runs)
python3 -c "
import json, glob
results = []
for f in sorted(glob.glob('$OUTDIR/*_summary.json')):
    d = json.load(open(f))
    results.append(d)
results.sort(key=lambda x: -x['accuracy'])
print(f\"{'Config':<50} {'Acc':>6} {'N':>4} {'Time':>7}\")
print('-'*70)
for d in results:
    print(f\"{d['tag']:<50} {d['accuracy']:>6.4f} {d['total']:>4} {d['time_s']:>7.0f}s\")
"

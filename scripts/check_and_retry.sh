#!/usr/bin/env bash
# Check best_eval jobs: if a shard has no result and its log file hasn't been
# updated in 30+ minutes, kill and resubmit with bsz=1.
set -euo pipefail
cd "$(dirname "$0")/.."

VOLC=~/.volc/bin/volc
TMP=/tmp/_retry.yml
TAG="lowprob_t0.3_c1_r0.25"
NOW=$(date +%s)
STALE_SECS=1800  # 30 minutes

declare -A DS_TOTAL=(
  [humaneval]=164 [ifeval]=541 [aime2025]=30 [gpqa]=198
  [triviaqa]=17944 [drop]=9535 [hellaswag]=10042 [piqa]=1838 [cmath]=1098
)
declare -A DS_BSZ=(
  [humaneval]=1 [ifeval]=1 [aime2025]=1 [gpqa]=1
  [triviaqa]=8 [drop]=8 [hellaswag]=8 [piqa]=8 [cmath]=8
)

MAX_PER_SHARD=64
RETRIED=0
DONE=0
RUNNING=0

for DATASET in humaneval ifeval aime2025 gpqa triviaqa drop hellaswag piqa cmath; do
  TOTAL=${DS_TOTAL[$DATASET]}
  BSZ=${DS_BSZ[$DATASET]}
  NUM_SHARDS=$(( (TOTAL + MAX_PER_SHARD - 1) / MAX_PER_SHARD ))
  OUT_DIR="results_v2/best_eval/${DATASET}/${TAG}"

  for ((S=0; S<NUM_SHARDS; S++)); do
    # Check if both original and remask summaries exist
    ORIG_DONE=false
    REMASK_DONE=false
    [ -f "${OUT_DIR}/original_shard${S}_summary.json" ] && ORIG_DONE=true
    for f in "${OUT_DIR}"/remask_*_shard${S}_summary.json; do
      [ -f "$f" ] && REMASK_DONE=true
      break
    done

    if $ORIG_DONE && $REMASK_DONE; then
      # Verify: summary says done=true and jsonl has >0 lines
      BAD=false
      for SF in "${OUT_DIR}"/original_shard${S}_summary.json "${OUT_DIR}"/remask_*_shard${S}_summary.json; do
        [ ! -f "$SF" ] && continue
        IS_DONE=$(python3 -c "import json; print(json.load(open('$SF')).get('done', False))" 2>/dev/null || echo "False")
        STOTAL=$(python3 -c "import json; print(json.load(open('$SF')).get('total', 0))" 2>/dev/null || echo "0")
        if [ "$IS_DONE" != "True" ] || [ "$STOTAL" -lt 1 ]; then
          BAD=true
          echo "[INCOMPLETE] ${DATASET} shard ${S}: $(basename $SF) done=${IS_DONE} total=${STOTAL}"
        fi
      done
      if ! $BAD; then
        DONE=$((DONE + 1))
        continue
      fi
      # Fall through to stale check
    fi

    # Check log freshness
    LOGFILE="${OUT_DIR}/original_shard${S}.log"
    [ ! -f "$LOGFILE" ] && LOGFILE="${OUT_DIR}/remask_shard${S}.log"

    STALE=false
    if [ -f "$LOGFILE" ]; then
      MTIME=$(stat -c %Y "$LOGFILE" 2>/dev/null || echo 0)
      AGE=$((NOW - MTIME))
      if [ $AGE -gt $STALE_SECS ]; then
        STALE=true
      fi
    else
      # No log at all - check if output dir exists and is old
      if [ -d "$OUT_DIR" ]; then
        DIR_MTIME=$(stat -c %Y "$OUT_DIR" 2>/dev/null || echo 0)
        DIR_AGE=$((NOW - DIR_MTIME))
        if [ $DIR_AGE -gt $STALE_SECS ]; then
          # Dir exists but no log for 30+ min - task likely never ran
          STALE=true
          AGE=$DIR_AGE
        else
          RUNNING=$((RUNNING + 1))
          continue
        fi
      else
        # No dir, no log - not submitted yet or just queued
        RUNNING=$((RUNNING + 1))
        continue
      fi
    fi

    if ! $STALE; then
      # Check for crash signals in log
      CRASHED=false
      if [ -f "$LOGFILE" ]; then
        if grep -qiE "Error|Traceback|OOM|OutOfMemory|CUDA|killed|FAILED|exit code" "$LOGFILE" 2>/dev/null; then
          CRASHED=true
          echo "[CRASHED] ${DATASET} shard ${S}: error found in log"
        fi
      fi
      if ! $CRASHED; then
        RUNNING=$((RUNNING + 1))
        continue
      fi
      # Fall through to kill + resubmit
      AGE=0
    fi

    # Stale: try to kill old task first
    for OLD_NAME in "best-${DATASET}-s${S}of${NUM_SHARDS}" "retry-${DATASET}-s${S}"; do
      OLD_CLEAN=$(echo "$OLD_NAME" | tr '.' 'd' | head -c 50)
      $VOLC ml_task stop -n "$OLD_CLEAN" 2>/dev/null || true
    done
    sleep 0.5

    # Resubmit with bsz=1
    echo "[RETRY] ${DATASET} shard ${S}: stale (${AGE}s), killed old task, resubmitting with bsz=1"

    TASK_NAME="retry-${DATASET}-s${S}-bsz1"
    TASK_NAME=$(echo "$TASK_NAME" | tr '.' 'd' | head -c 60)

    cat > "$TMP" << EOF
TaskName: "${TASK_NAME}"
Storages:
  - Type: "Vepfs"
    MountPath: "/vepfs-mlp2/c20250506/251105017"
    SubPath: "/c20250506/251105017"
Entrypoint: "cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask && export HF_TOKEN=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 && bash scripts/run_best_eval_single.sh ${DATASET} ${S} ${NUM_SHARDS} 1"
Tags: [LLM, yl, llada, remask, retry]
ImageUrl: "cr-mlp-cn-beijing.cr.volces.com/public/yl_dllm:1.0"
Framework: "Custom"
ResourceQueueName: "c20250506"
TaskRoleSpecs:
  - RoleName: "worker"
    RoleReplicas: 1
    Flavor: "ml.pni2.3xlarge"
ActiveDeadlineSeconds: 10800
DelayExitTimeSeconds: 0
AccessType: "Private"
Preemptible: true
Priority: 6
EOF

    $VOLC ml_task submit -c "$TMP" 2>&1 | grep -oE "TaskId[^,]*" || echo "  submitted"
    RETRIED=$((RETRIED + 1))
    sleep 0.2
  done
done

rm -f "$TMP"

TOTAL_SHARDS=$((DONE + RUNNING + RETRIED))
echo ""
echo "=== Summary ==="
echo "  Done:    ${DONE}"
echo "  Running: ${RUNNING}"
echo "  Retried: ${RETRIED} (with bsz=1)"
echo "  Total:   ${TOTAL_SHARDS} shards"

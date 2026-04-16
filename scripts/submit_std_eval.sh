#!/usr/bin/env bash
# Submit HumanEval + MBPP with EvalPlus standard prompt.
#   Both use the same EvalPlus prompt format (zero-shot):
#     "Please provide a self-contained Python script..."
#   This is the exact format used by LLaMA 3.1, Qwen2.5-Coder,
#   DeepSeek-Coder V2 on the EvalPlus leaderboard.
#
# Same hyper-params as best_eval: LowProb τ=0.3, C=1, ρ=0.25
# Shard size ≤ 32, single GPU per job.
set -euo pipefail
cd "$(dirname "$0")/.."

VOLC=~/.volc/bin/volc
TMP=/tmp/_std_eval.yml

TAG="lowprob_t0.3_c1_r0.25"

declare -A DS_TOTAL=(
  [humaneval]=164
  [mbpp]=378
)

MAX_PER_SHARD=32
SUBMITTED=0

for DATASET in humaneval mbpp; do
  TOTAL=${DS_TOTAL[$DATASET]}
  NUM_SHARDS=$(( (TOTAL + MAX_PER_SHARD - 1) / MAX_PER_SHARD ))

  echo "=== ${DATASET}_std: ${TOTAL} samples, ${NUM_SHARDS} shards ==="

  for ((S=0; S<NUM_SHARDS; S++)); do
    for MODE in original remask; do
      TASK_NAME="std-${DATASET}-${MODE}-s${S}of${NUM_SHARDS}"
      TASK_NAME=$(echo "$TASK_NAME" | tr '.' 'd' | head -c 60)

      OUT_DIR="results_v2/best_eval_std/${DATASET}/${TAG}"
      if [ "$MODE" = "original" ]; then
        ls "${OUT_DIR}"/original*shard${S}_summary.json &>/dev/null && { echo "  [$S] $MODE SKIP"; continue; }
      else
        ls "${OUT_DIR}"/remask_*shard${S}_summary.json &>/dev/null && { echo "  [$S] $MODE SKIP"; continue; }
      fi

      cat > "$TMP" << EOF
TaskName: "${TASK_NAME}"
Storages:
  - Type: "Vepfs"
    MountPath: "/vepfs-mlp2/c20250506/251105017"
    SubPath: "/c20250506/251105017"
Entrypoint: "cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask && export HF_TOKEN=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 && bash scripts/run_std_eval_single.sh ${DATASET} ${S} ${NUM_SHARDS} ${MODE} 1"
Tags: [LLM, yl, llada, remask, std-eval]
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
RetryOptions:
    EnableRetry: true
    MaxRetryTimes: 5
    IntervalSeconds: 120
    PolicySets:
        - "Failed"
        - "InstanceReclaimed"
EOF

      echo -n "  [$S] ${TASK_NAME} ... "
      $VOLC ml_task submit -c "$TMP" 2>&1 | grep -oE "task_id=[^ ]*" || echo "ok"
      sleep 0.2
      SUBMITTED=$((SUBMITTED + 1))
    done
  done
done

rm -f "$TMP"
echo ""
echo "Submitted ${SUBMITTED} jobs total."

#!/usr/bin/env bash
# Submit standard-aligned evaluations.
#
# All benchmarks aligned with LLaMA 3.1 / lm-eval-harness / EvalPlus:
#   humaneval  - EvalPlus zero-shot, gen=768
#   mbpp       - EvalPlus zero-shot, gen=768
#   bbh        - 3-shot CoT (BIG-Bench-Hard), gen=1024
#   mmlu_pro   - 5-shot CoT per-category, gen=2048
#   drop       - 3-shot, gen=256
#   triviaqa   - 5-shot, gen=128
#
# Hyper-params: LowProb τ=0.3, C=1, ρ=0.25
# Shard ≤ 32, bsz=1, single GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

VOLC=~/.volc/bin/volc
TMP=/tmp/_std_eval.yml

TAG="lowprob_t0.3_c1_r0.25"
MAX_PER_SHARD=32

declare -A DS_TOTAL=(
  [humaneval]=164
  [mbpp]=378
  [bbh]=6511
  [mmlu_pro]=12032
  [drop]=9536
  [triviaqa]=17944
)

declare -A DS_TIMEOUT=(
  [humaneval]=10800
  [mbpp]=10800
  [bbh]=14400
  [mmlu_pro]=21600
  [drop]=10800
  [triviaqa]=10800
)

SUBMITTED=0

for DATASET in humaneval mbpp bbh mmlu_pro drop triviaqa; do
  TOTAL=${DS_TOTAL[$DATASET]}
  TIMEOUT=${DS_TIMEOUT[$DATASET]}
  NUM_SHARDS=$(( (TOTAL + MAX_PER_SHARD - 1) / MAX_PER_SHARD ))

  echo "=== ${DATASET}: ${TOTAL} samples, ${NUM_SHARDS} shards, timeout=${TIMEOUT}s ==="

  for ((S=0; S<NUM_SHARDS; S++)); do
    for MODE in original remask; do
      TASK_NAME="std-${DATASET}-${MODE}-s${S}of${NUM_SHARDS}"
      TASK_NAME=$(echo "$TASK_NAME" | tr '.' 'd' | head -c 60)

      OUT_DIR="results_std/${DATASET}/${TAG}"
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
ActiveDeadlineSeconds: ${TIMEOUT}
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

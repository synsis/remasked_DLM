#!/usr/bin/env bash
# Submit v5 eval: lowprob τ=0.3, C=3, ρ=0.25 → results_v5
# Datasets: aime2025, gsm_plus, humaneval, piqa (same as results_v3)
# Timeout: 24h, Retry: 5 times on Failed / InstanceReclaimed
set -euo pipefail
cd "$(dirname "$0")/.."

VOLC=~/.volc/bin/volc
TMP=/tmp/_submit_v5.yml

declare -A DS_TOTAL=(
  [aime2025]=30
  [gsm_plus]=2400
  [humaneval]=164
  [piqa]=1838
)

MAX_PER_SHARD=32
SUBMITTED=0

for DATASET in aime2025 gsm_plus humaneval piqa; do
  TOTAL=${DS_TOTAL[$DATASET]}
  NUM_SHARDS=$(( (TOTAL + MAX_PER_SHARD - 1) / MAX_PER_SHARD ))

  echo "=== ${DATASET}: ${TOTAL} samples, ${NUM_SHARDS} shards ==="

  for ((S=0; S<NUM_SHARDS; S++)); do
    for MODE in original remask; do
      TASK_NAME="v5-${DATASET}-${MODE}-s${S}of${NUM_SHARDS}"
      TASK_NAME=$(echo "$TASK_NAME" | tr '.' 'd' | head -c 60)

      cat > "$TMP" << EOF
TaskName: "${TASK_NAME}"
Storages:
  - Type: "Vepfs"
    MountPath: "/vepfs-mlp2/c20250506/251105017"
    SubPath: "/c20250506/251105017"
Entrypoint: "cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask && export HF_TOKEN=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 && bash scripts/run_v5_single.sh ${DATASET} ${S} ${NUM_SHARDS} ${MODE} 1"
Tags: [LLM, yl, llada, remask, v5]
ImageUrl: "cr-mlp-cn-beijing.cr.volces.com/public/yl_dllm:1.0"
Framework: "Custom"
ResourceQueueName: "c20250506"
TaskRoleSpecs:
  - RoleName: "worker"
    RoleReplicas: 1
    Flavor: "ml.pni2.3xlarge"
ActiveDeadlineSeconds: 86400
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

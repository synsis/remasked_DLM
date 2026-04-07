#!/usr/bin/env bash
# Submit v6 extra: τ=0.5, ρ=0.25, C=1/3/5 on AIME, remask only
# Note: c3/r0.25 was tested in v4, but c1 and c5 with r0.25 are new
set -euo pipefail
cd "$(dirname "$0")/.."

VOLC=~/.volc/bin/volc
TMP=/tmp/_submit_v6_extra.yml
SUBMITTED=0

DATASET="aime2025"
NUM_SHARDS=1
SHARD=0
MODE="remask"

declare -a CONFIGS=(
  "0.5 1 0.25"
  "0.5 3 0.25"
  "0.5 5 0.25"
)

echo "=== v6-extra: τ=0.5 ρ=0.25 C=1/3/5 on ${DATASET} (remask only) ==="

for CFG in "${CONFIGS[@]}"; do
  read -r TAU C_MAX RHO <<< "$CFG"
  TAG="lowprob_t${TAU}_c${C_MAX}_r${RHO}"
  TASK_NAME="v6-${DATASET}-${TAG}"
  TASK_NAME=$(echo "$TASK_NAME" | tr '.' 'd' | head -c 60)

  cat > "$TMP" << EOF
TaskName: "${TASK_NAME}"
Storages:
  - Type: "Vepfs"
    MountPath: "/vepfs-mlp2/c20250506/251105017"
    SubPath: "/c20250506/251105017"
Entrypoint: "cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask && export HF_TOKEN=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 && bash scripts/run_v6_single.sh ${DATASET} ${SHARD} ${NUM_SHARDS} ${MODE} 1 ${TAU} ${C_MAX} ${RHO}"
Tags: [LLM, yl, llada, remask, v6]
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

  echo -n "  [${SUBMITTED}] τ=${TAU} C=${C_MAX} ρ=${RHO} (${TASK_NAME}) ... "
  $VOLC ml_task submit -c "$TMP" 2>&1 | grep -oE "task_id=[^ ]*" || echo "ok"
  sleep 0.2
  SUBMITTED=$((SUBMITTED + 1))
done

rm -f "$TMP"
echo ""
echo "Submitted ${SUBMITTED} jobs total."

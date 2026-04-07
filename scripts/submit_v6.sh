#!/usr/bin/env bash
# Submit v6: 30 low_prob configs on AIME only, remask only (no original)
# Picks top-30 from CMATH ablation, excluding 3 already tested (v3/v4/v5)
# and 12 worst performers
set -euo pipefail
cd "$(dirname "$0")/.."

VOLC=~/.volc/bin/volc
TMP=/tmp/_submit_v6.yml
SUBMITTED=0

DATASET="aime2025"
TOTAL=30
NUM_SHARDS=1
SHARD=0
MODE="remask"

declare -a CONFIGS=(
  # τ=0.1: all 9 (all 91% on CMATH)
  "0.1 1 0.25"
  "0.1 1 0.5"
  "0.1 1 1.0"
  "0.1 3 0.25"
  "0.1 3 0.5"
  "0.1 3 1.0"
  "0.1 5 0.25"
  "0.1 5 0.5"
  "0.1 5 1.0"
  # τ=0.3: 8 (excl c3/r0.25 = v5)
  "0.3 1 0.25"   # 93%
  "0.3 1 0.5"    # 93%
  "0.3 1 1.0"    # 93%
  "0.3 3 0.5"    # 92%
  "0.3 3 1.0"    # 92%
  "0.3 5 0.25"   # 91%
  "0.3 5 0.5"    # 92%
  "0.3 5 1.0"    # 92%
  # τ=0.5: 4 (excl c3/r0.25=v4, c1/r0.25=86%, c1/r1.0=90%, c5/r0.25=90%, c3/r1.0=91%)
  "0.5 1 0.5"    # 91%
  "0.5 3 0.5"    # 91%
  "0.5 5 0.5"    # 92%
  "0.5 5 1.0"    # 92%
  # τ=0.7: 6 (excl c3/r1.0=v3, c1/r0.5=90%, c1/r1.0=88%)
  "0.7 1 0.25"   # 93%
  "0.7 3 0.25"   # 91%
  "0.7 3 0.5"    # 94%
  "0.7 5 0.25"   # 91%
  "0.7 5 0.5"    # 92%
  "0.7 5 1.0"    # 94%
  # τ=0.9: 3 (excl the aggressive/weak ones ≤90%)
  "0.9 1 0.25"   # 93%
  "0.9 1 0.5"    # 92%
  "0.9 3 0.25"   # 93%
)

echo "=== v6: ${#CONFIGS[@]} low_prob configs on ${DATASET} (remask only) ==="

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

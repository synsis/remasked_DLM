#!/usr/bin/env bash
# Submit bisect jobs to identify AIME2025 regression
# 3 test configs × 2 modes (original + remask) = 6 jobs
set -euo pipefail
cd "$(dirname "$0")/.."

VOLC=~/.volc/bin/volc
TMP=/tmp/_submit_bisect.yml
SUBMITTED=0

for TEST_ID in A_old_code B_lazy_mask C_head_oldprompt; do
  for MODE in original remask; do
    TASK_NAME="bisect-${TEST_ID}-${MODE}"
    TASK_NAME=$(echo "$TASK_NAME" | tr '.' 'd' | tr '_' '-' | head -c 60)

    cat > "$TMP" << EOF
TaskName: "${TASK_NAME}"
Storages:
  - Type: "Vepfs"
    MountPath: "/vepfs-mlp2/c20250506/251105017"
    SubPath: "/c20250506/251105017"
Entrypoint: "cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask && export HF_TOKEN=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 && bash scripts/run_bisect_single.sh ${TEST_ID} ${MODE}"
Tags: [LLM, yl, llada, remask, bisect]
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

    echo -n "  ${TASK_NAME} ... "
    $VOLC ml_task submit -c "$TMP" 2>&1 | grep -oE "task_id=[^ ]*" || echo "ok"
    sleep 0.2
    SUBMITTED=$((SUBMITTED + 1))
  done
done

rm -f "$TMP"
echo ""
echo "Submitted ${SUBMITTED} bisect jobs total."
echo ""
echo "Tests:"
echo "  A_old_code       = commit 0b3b001 (pre-computed attn mask, old #### prompt)"
echo "  B_lazy_mask      = commit 39d7bac (lazy attn mask, new Answer: prompt)"
echo "  C_head_oldprompt = current HEAD (lazy attn mask, old #### prompt)"
echo "  [baseline]       = current HEAD (lazy attn mask, new Answer: prompt) = v4/v6 results (33.33%)"
echo ""
echo "All tests use: tau=0.5, C=3, rho=0.25, gen_length=16384"

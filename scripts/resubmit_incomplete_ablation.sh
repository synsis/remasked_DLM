#!/usr/bin/env bash
# Resubmit 8 incomplete ablation configs (100 sample CMATH).
# Each config ran but didn't complete all 100 samples (preempted).
set -euo pipefail
cd "$(dirname "$0")/.."

VOLC=~/.volc/bin/volc
TMP_YML="/tmp/_resub_abl.yml"
SUBMITTED=0

submit_one() {
  local DIR_NAME=$1 STRATEGY=$2 TAU=$3 C_MAX=$4 RHO=$5

  local OUT_DIR="results_v2/ablation/${DIR_NAME}"
  rm -f "${OUT_DIR}"/*_summary.json "${OUT_DIR}"/*_results.jsonl "${OUT_DIR}"/run.log
  echo "  Cleaned ${OUT_DIR}"

  local TASK_NAME="abl-redo-${DIR_NAME}"
  TASK_NAME=$(echo "$TASK_NAME" | tr '.' 'd' | head -c 60)

  local ENTRY="cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask && "
  ENTRY+="export HF_TOKEN=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 && "
  ENTRY+="export https_proxy=100.68.162.212:3128 && "
  ENTRY+="export http_proxy=100.68.162.212:3128 && "
  ENTRY+="conda run -n remask python -u -m eval.cmath "
  ENTRY+="--output_dir ${OUT_DIR} --mode remask --batch_size 1 --max_samples 100 --sample_seed 42 "
  ENTRY+="--strategy ${STRATEGY} --remask_threshold ${TAU} --max_remask_per_pos ${C_MAX} --max_remask_ratio ${RHO} "
  ENTRY+="2>&1 | tee ${OUT_DIR}/run.log"

  cat > "$TMP_YML" << EOF
TaskName: "${TASK_NAME}"
Storages:
  - Type: "Vepfs"
    MountPath: "/vepfs-mlp2/c20250506/251105017"
    SubPath: "/c20250506/251105017"
Entrypoint: "${ENTRY}"
Tags: [LLM, yl, llada, remask, ablation, redo]
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

  echo -n "  ${TASK_NAME} ... "
  $VOLC ml_task submit -c "$TMP_YML" 2>&1 | grep -oE "TaskId[^,]*" || echo "ok"
  sleep 0.3
  SUBMITTED=$((SUBMITTED + 1))
}

echo "=== Resubmitting 8 incomplete ablation configs ==="
echo ""

#                DIR_NAME                       STRATEGY     TAU  C  RHO
submit_one "016_low_prob_t0.5_c5_r0.25"      low_prob      0.5  5  0.25
submit_one "023_low_prob_t0.7_c3_r0.5"       low_prob      0.7  3  0.50
submit_one "034_low_prob_t0.9_c5_r0.25"      low_prob      0.9  5  0.25
submit_one "064_logit_diff_t0.1_c1_r0.25"    logit_diff    0.1  1  0.25
submit_one "080_logit_diff_t0.2_c5_r0.5"     logit_diff    0.2  5  0.50
submit_one "089_logit_diff_t0.3_c5_r0.5"     logit_diff    0.3  5  0.50
submit_one "097_logit_diff_t0.5_c5_r0.25"    logit_diff    0.5  5  0.25
submit_one "098_logit_diff_t0.5_c5_r0.5"     logit_diff    0.5  5  0.50

rm -f "$TMP_YML"
echo ""
echo "Done. Submitted: ${SUBMITTED}"

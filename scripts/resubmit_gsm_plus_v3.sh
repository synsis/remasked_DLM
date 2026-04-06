#!/usr/bin/env bash
# Resubmit 5 failed GSM Plus shards (best params, results_v3).
set -euo pipefail
cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask

VOLC=~/.volc/bin/volc
TMP=/tmp/_resub_gsm_v3.yml
OUTDIR="results_v3/gsm_plus/lowprob_t0.7_c3_r1.0"
NSHARDS=75
SUBMITTED=0

FAILED_SHARDS=(19 20 32 69 71)

submit_one() {
  local SHARD=$1
  local TASK_NAME="v3-gsm-remask-s${SHARD}of${NSHARDS}"

  cat > "$TMP" << EOF
TaskName: "${TASK_NAME}"
Storages:
  - Type: "Vepfs"
    MountPath: "/vepfs-mlp2/c20250506/251105017"
    SubPath: "/c20250506/251105017"
Entrypoint: "cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask && export HF_TOKEN=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 && bash scripts/run_v3_best_single.sh gsm_plus ${SHARD} ${NSHARDS} remask 1"
Tags: [LLM, yl, llada, remask, v3-best, gsm-plus]
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
EOF

  echo -n "  ${TASK_NAME} ... "
  $VOLC ml_task submit -c "$TMP" 2>&1 | grep -oE "task_id=[^ ]*" || echo "ok"
  sleep 0.2
  SUBMITTED=$((SUBMITTED + 1))
}

echo "=== Deleting incomplete files for failed shards ==="
for S in "${FAILED_SHARDS[@]}"; do
  echo "  Removing shard $S ..."
  rm -f "${OUTDIR}/remask_bsz1_shard${S}.log"
  rm -f "${OUTDIR}/remask_low_prob_0.7_r1.0_bsz1_shard${S}_results.jsonl"
  rm -f "${OUTDIR}/remask_low_prob_0.7_r1.0_bsz1_shard${S}_summary.json"
done

echo ""
echo "=== Submitting ${#FAILED_SHARDS[@]} shards ==="
for S in "${FAILED_SHARDS[@]}"; do
  submit_one "$S"
done

rm -f "$TMP"
echo ""
echo "Total submitted: ${SUBMITTED}"

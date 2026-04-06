#!/usr/bin/env bash
# Resubmit 6 incomplete mmlu_pro shards with 24h timeout.
#   original: 140, 280, 293, 297
#   remask:   296, 307
set -euo pipefail
cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask

VOLC=~/.volc/bin/volc
TMP=/tmp/_resub_mmlupro.yml
OUTDIR="results_v2/best_eval/mmlu_pro/lowprob_t0.3_c1_r0.25"
NSHARDS=376
SUBMITTED=0

submit_one() {
  local MODE=$1 SHARD=$2
  local TASK_NAME="re24h-mmlupro-${MODE}-s${SHARD}of${NSHARDS}"
  TASK_NAME=$(echo "$TASK_NAME" | tr '.' 'd' | head -c 60)

  cat > "$TMP" << EOF
TaskName: "${TASK_NAME}"
Storages:
  - Type: "Vepfs"
    MountPath: "/vepfs-mlp2/c20250506/251105017"
    SubPath: "/c20250506/251105017"
Entrypoint: "cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask && export HF_TOKEN=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 && bash scripts/run_best_eval_single.sh mmlu_pro ${SHARD} ${NSHARDS} ${MODE} 1"
Tags: [LLM, yl, llada, remask, best-eval, retry-24h]
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

echo "=== Deleting incomplete files ==="
for S in 140 280 293 297; do
  echo "  Removing original shard $S ..."
  rm -f "${OUTDIR}/original_bsz1_shard${S}.log"
  rm -f "${OUTDIR}/original_bsz1_shard${S}_results.jsonl"
  rm -f "${OUTDIR}/original_bsz1_shard${S}_summary.json"
done
for S in 296 307; do
  echo "  Removing remask shard $S ..."
  rm -f "${OUTDIR}/remask_bsz1_shard${S}.log"
  rm -f "${OUTDIR}/remask_low_prob_0.3_bsz1_shard${S}_results.jsonl"
  rm -f "${OUTDIR}/remask_low_prob_0.3_bsz1_shard${S}_summary.json"
done

echo ""
echo "=== Submitting 6 jobs (timeout=24h) ==="
for S in 140 280 293 297; do
  submit_one original "$S"
done
for S in 296 307; do
  submit_one remask "$S"
done

rm -f "$TMP"
echo ""
echo "Total submitted: ${SUBMITTED}"

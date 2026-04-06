#!/usr/bin/env bash
# Resubmit incomplete bbh shards (best_eval1) with 24h timeout.
# Excludes remask shard 151 per user request.
set -euo pipefail
cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask

VOLC=~/.volc/bin/volc
TMP=/tmp/_resub_bbh.yml
OUTDIR="results_v2/best_eval1/bbh/lowprob_t0.3_c1_r0.25"
NSHARDS=204
SUBMITTED=0

submit_one() {
  local MODE=$1 SHARD=$2
  local TASK_NAME="re24h-bbh-${MODE}-s${SHARD}of${NSHARDS}"
  TASK_NAME=$(echo "$TASK_NAME" | tr '.' 'd' | head -c 60)

  cat > "$TMP" << EOF
TaskName: "${TASK_NAME}"
Storages:
  - Type: "Vepfs"
    MountPath: "/vepfs-mlp2/c20250506/251105017"
    SubPath: "/c20250506/251105017"
Entrypoint: "cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask && export HF_TOKEN=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 && bash scripts/run_best_eval1_single.sh bbh ${SHARD} ${NSHARDS} ${MODE} 1"
Tags: [LLM, yl, llada, remask, best-eval1, retry-24h]
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

# --- Original: delete not-done, then submit missing + not-done ---
ORIG_NOTDONE=(130 143 144 164)
ORIG_MISSING=(147 172 174 189)

echo "=== Deleting incomplete original files ==="
for S in "${ORIG_NOTDONE[@]}"; do
  echo "  Removing original shard $S ..."
  rm -f "${OUTDIR}/original_bsz1_shard${S}.log"
  rm -f "${OUTDIR}/original_bsz1_shard${S}_results.jsonl"
  rm -f "${OUTDIR}/original_bsz1_shard${S}_summary.json"
done

echo ""
echo "=== Submitting original shards (${#ORIG_MISSING[@]} missing + ${#ORIG_NOTDONE[@]} not-done = $((${#ORIG_MISSING[@]} + ${#ORIG_NOTDONE[@]}))) ==="
for S in "${ORIG_MISSING[@]}" "${ORIG_NOTDONE[@]}"; do
  submit_one original "$S"
done

# --- Remask: delete not-done (except 151), then submit missing + not-done ---
REM_NOTDONE=(12 26 114 118 119)  # 151 excluded
REM_MISSING=(18 24 25 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84)

echo ""
echo "=== Deleting incomplete remask files ==="
for S in "${REM_NOTDONE[@]}"; do
  echo "  Removing remask shard $S ..."
  rm -f "${OUTDIR}/remask_bsz1_shard${S}.log"
  rm -f "${OUTDIR}/remask_low_prob_0.3_bsz1_shard${S}_results.jsonl"
  rm -f "${OUTDIR}/remask_low_prob_0.3_bsz1_shard${S}_summary.json"
done

echo ""
echo "=== Submitting remask shards (${#REM_MISSING[@]} missing + ${#REM_NOTDONE[@]} not-done = $((${#REM_MISSING[@]} + ${#REM_NOTDONE[@]}))) ==="
for S in "${REM_MISSING[@]}" "${REM_NOTDONE[@]}"; do
  submit_one remask "$S"
done

rm -f "$TMP"
echo ""
echo "Total submitted: ${SUBMITTED}"

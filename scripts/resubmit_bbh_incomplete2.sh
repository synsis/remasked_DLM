#!/usr/bin/env bash
# Resubmit incomplete bbh shards (best_eval1), round 2.
# Excludes remask shard 151.
set -euo pipefail
cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask

VOLC=~/.volc/bin/volc
TMP=/tmp/_resub_bbh2.yml
OUTDIR="results_v2/best_eval1/bbh/lowprob_t0.3_c1_r0.25"
NSHARDS=204
SUBMITTED=0

submit_one() {
  local MODE=$1 SHARD=$2
  local TASK_NAME="r2-bbh-${MODE}-s${SHARD}of${NSHARDS}"
  TASK_NAME=$(echo "$TASK_NAME" | tr '.' 'd' | head -c 60)

  cat > "$TMP" << EOF
TaskName: "${TASK_NAME}"
Storages:
  - Type: "Vepfs"
    MountPath: "/vepfs-mlp2/c20250506/251105017"
    SubPath: "/c20250506/251105017"
Entrypoint: "cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask && export HF_TOKEN=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 && bash scripts/run_best_eval1_single.sh bbh ${SHARD} ${NSHARDS} ${MODE} 1"
Tags: [LLM, yl, llada, remask, best-eval1, retry-24h, r2]
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

# --- Original not-done: 164, 172, 189 ---
ORIG_NOTDONE=(164 172 189)

echo "=== Deleting incomplete original files ==="
for S in "${ORIG_NOTDONE[@]}"; do
  echo "  Removing original shard $S ..."
  rm -f "${OUTDIR}/original_bsz1_shard${S}.log"
  rm -f "${OUTDIR}/original_bsz1_shard${S}_results.jsonl"
  rm -f "${OUTDIR}/original_bsz1_shard${S}_summary.json"
done

echo ""
echo "=== Submitting original (${#ORIG_NOTDONE[@]} not-done) ==="
for S in "${ORIG_NOTDONE[@]}"; do
  submit_one original "$S"
done

# --- Remask missing + not-done (excluding 151) ---
REM_MISSING=(12 18 26 33 42 56 69 70 71 72 73 74 77 78 79 80 81 82 83 84 114 118 119)
REM_NOTDONE=(24 25 28 31 34 35 38 39 40 58 60 65 66 67 68 75 76)

echo ""
echo "=== Deleting incomplete remask files ==="
for S in "${REM_NOTDONE[@]}"; do
  echo "  Removing remask shard $S ..."
  rm -f "${OUTDIR}/remask_bsz1_shard${S}.log"
  rm -f "${OUTDIR}/remask_low_prob_0.3_bsz1_shard${S}_results.jsonl"
  rm -f "${OUTDIR}/remask_low_prob_0.3_bsz1_shard${S}_summary.json"
done

echo ""
echo "=== Submitting remask (${#REM_MISSING[@]} missing + ${#REM_NOTDONE[@]} not-done = $((${#REM_MISSING[@]} + ${#REM_NOTDONE[@]}))) ==="
for S in "${REM_MISSING[@]}" "${REM_NOTDONE[@]}"; do
  submit_one remask "$S"
done

rm -f "$TMP"
echo ""
echo "Total submitted: ${SUBMITTED}"

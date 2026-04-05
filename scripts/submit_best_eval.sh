#!/usr/bin/env bash
# Submit all dataset evaluations with "most efficient" remask setting.
# Each job = 1 GPU, 1 shard, 1 mode (original OR remask).
set -euo pipefail
cd "$(dirname "$0")/.."

VOLC=~/.volc/bin/volc
TMP=/tmp/_best_eval.yml

TAG="lowprob_t0.3_c1_r0.25"

declare -A DS_TOTAL=(
  [humaneval]=164 [ifeval]=541 [aime2025]=30 [gpqa]=198
  [triviaqa]=17944 [drop]=9535 [hellaswag]=10042 [piqa]=1838 [cmath]=1098
  [bbh]=6511 [gsm_plus]=2400 [mmlu_pro]=12032
)

MAX_PER_SHARD=32
SUBMITTED=0

for DATASET in humaneval ifeval aime2025 gpqa triviaqa drop hellaswag piqa cmath bbh gsm_plus mmlu_pro; do
  TOTAL=${DS_TOTAL[$DATASET]}
  NUM_SHARDS=$(( (TOTAL + MAX_PER_SHARD - 1) / MAX_PER_SHARD ))

  echo "=== ${DATASET}: ${TOTAL} samples, ${NUM_SHARDS} shards ==="

  for ((S=0; S<NUM_SHARDS; S++)); do
    for MODE in original remask; do
      TASK_NAME="${DATASET}-${MODE}-s${S}of${NUM_SHARDS}"
      TASK_NAME=$(echo "$TASK_NAME" | tr '.' 'd' | head -c 60)

      # Skip if already done (match any bsz or old naming)
      OUT_DIR="results_v2/best_eval/${DATASET}/${TAG}"
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
Entrypoint: "cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask && export HF_TOKEN=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 && bash scripts/run_best_eval_single.sh ${DATASET} ${S} ${NUM_SHARDS} ${MODE} 1"
Tags: [LLM, yl, llada, remask, best-eval]
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

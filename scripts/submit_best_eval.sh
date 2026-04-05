#!/usr/bin/env bash
# Submit all dataset evaluations with "most efficient" remask setting.
# Each job = 1 GPU, 1 shard (max 64 samples), runs both original + remask.
set -euo pipefail
cd "$(dirname "$0")/.."

VOLC=~/.volc/bin/volc
TMP=/tmp/_best_eval.yml

TAG="lowprob_t0.3_c1_r0.25"

# Dataset configs: name, total_samples, bsz, max_per_shard
declare -A DS_TOTAL=(
  [humaneval]=164 [ifeval]=541 [aime2025]=30 [gpqa]=198
  [triviaqa]=17944 [drop]=9535 [hellaswag]=10042 [piqa]=1838 [cmath]=1098
  [bbh]=6511 [gsm_plus]=1319 [mmlu_pro]=12032
)
declare -A DS_BSZ=(
  [humaneval]=1 [ifeval]=1 [aime2025]=1 [gpqa]=1
  [triviaqa]=8 [drop]=8 [hellaswag]=8 [piqa]=8 [cmath]=8
  [bbh]=8 [gsm_plus]=1 [mmlu_pro]=8
)

MAX_PER_SHARD=64
SUBMITTED=0

for DATASET in humaneval ifeval aime2025 gpqa triviaqa drop hellaswag piqa cmath bbh gsm_plus mmlu_pro; do
  TOTAL=${DS_TOTAL[$DATASET]}
  BSZ=${DS_BSZ[$DATASET]}
  NUM_SHARDS=$(( (TOTAL + MAX_PER_SHARD - 1) / MAX_PER_SHARD ))

  echo "=== ${DATASET}: ${TOTAL} samples, ${NUM_SHARDS} shards, bsz=${BSZ} ==="

  for ((S=0; S<NUM_SHARDS; S++)); do
    TASK_NAME="best-${DATASET}-s${S}of${NUM_SHARDS}-${TAG}"
    # Truncate task name to 60 chars, replace dots
    TASK_NAME=$(echo "$TASK_NAME" | tr '.' 'd' | head -c 60)

    # Skip if already done
    OUT_DIR="results_v2/best_eval/${DATASET}/${TAG}"
    if [ -f "${OUT_DIR}/remask_low_prob_0.3_c1_r0.25_shard${S}_summary.json" ] && \
       [ -f "${OUT_DIR}/original_shard${S}_summary.json" ]; then
      echo "  [$S] SKIP (done)"
      continue
    fi

    cat > "$TMP" << EOF
TaskName: "${TASK_NAME}"
Storages:
  - Type: "Vepfs"
    MountPath: "/vepfs-mlp2/c20250506/251105017"
    SubPath: "/c20250506/251105017"
Entrypoint: "cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask && export HF_TOKEN=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 && bash scripts/run_best_eval_single.sh ${DATASET} ${S} ${NUM_SHARDS} ${BSZ}"
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
    $VOLC ml_task submit -c "$TMP" 2>&1 | grep -oE "TaskId[^,]*" || echo "ok"
    sleep 0.2
    SUBMITTED=$((SUBMITTED + 1))
  done
done

rm -f "$TMP"
echo ""
echo "Submitted ${SUBMITTED} jobs total."

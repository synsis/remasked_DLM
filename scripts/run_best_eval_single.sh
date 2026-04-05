#!/usr/bin/env bash
# Run one shard of a dataset with the "most efficient" remask setting.
# Usage: bash run_best_eval_single.sh <dataset> <shard_id> <num_shards> <bsz>
#
# Most efficient setting: LowProb τ=0.3, C_max=1, ρ_max=0.25
set -euo pipefail
cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask

export https_proxy="100.68.162.212:3128"
export http_proxy="100.68.162.212:3128"
export HF_TOKEN="${HF_TOKEN:-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}"

DATASET=${1:?Usage: bash run_best_eval_single.sh <dataset> <shard_id> <num_shards> <bsz>}
SHARD_ID=${2:?}
NUM_SHARDS=${3:?}
BSZ=${4:-1}

# Best setting
STRATEGY="low_prob"
TAU="0.3"
C_MAX="1"
RHO="0.25"
TAG="lowprob_t${TAU}_c${C_MAX}_r${RHO}_bsz${BSZ}"

OUT_DIR="results_v2/best_eval/${DATASET}/${TAG}"
mkdir -p "$OUT_DIR"

echo "Dataset: ${DATASET}, Shard: ${SHARD_ID}/${NUM_SHARDS}, BSZ: ${BSZ}"
echo "Setting: ${TAG}"
echo "Output: ${OUT_DIR}"

# Run both original and remask
for MODE in original remask; do
  EXTRA=""
  EXTRA=""
  if [ "$MODE" = "remask" ]; then
    EXTRA="--strategy $STRATEGY --remask_threshold $TAU --max_remask_per_pos $C_MAX --max_remask_ratio $RHO"
  fi

  echo "  Running ${MODE}..."
  conda run -n remask python -u -m eval.${DATASET} \
    --output_dir "$OUT_DIR" \
    --mode "$MODE" \
    --batch_size "$BSZ" \
    --shard_id "$SHARD_ID" \
    --num_shards "$NUM_SHARDS" \
    $EXTRA \
    2>&1 | tee "${OUT_DIR}/${MODE}_shard${SHARD_ID}.log"
done

echo "Done: ${DATASET} shard ${SHARD_ID}"

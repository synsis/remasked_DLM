#!/usr/bin/env bash
# Run one shard with lowprob τ=0.5, C_max=3, ρ_max=0.25
# Usage: bash run_v4_single.sh <dataset> <shard_id> <num_shards> <mode> [bsz]
set -euo pipefail
cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask

export https_proxy="100.68.162.212:3128"
export http_proxy="100.68.162.212:3128"
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_ENDPOINT="https://huggingface.co"
export HUGGINGFACE_HUB_ENDPOINT="https://huggingface.co"

DATASET=${1:?Usage: bash run_v4_single.sh <dataset> <shard_id> <num_shards> <mode> [bsz]}
SHARD_ID=${2:?}
NUM_SHARDS=${3:?}
MODE=${4:?}  # original or remask
BSZ=${5:-1}

STRATEGY="low_prob"
TAU="0.5"
C_MAX="3"
RHO="0.25"
TAG="lowprob_t${TAU}_c${C_MAX}_r${RHO}"

OUT_DIR="results_v4/${DATASET}/${TAG}"
mkdir -p "$OUT_DIR"

echo "Dataset: ${DATASET}, Shard: ${SHARD_ID}/${NUM_SHARDS}, Mode: ${MODE}, BSZ: ${BSZ}"
echo "Output: ${OUT_DIR}"

EXTRA=""
if [ "$MODE" = "remask" ]; then
  EXTRA="--strategy $STRATEGY --remask_threshold $TAU --max_remask_per_pos $C_MAX --max_remask_ratio $RHO"
fi

conda run -n remask python -u -m eval.${DATASET} \
  --output_dir "$OUT_DIR" \
  --mode "$MODE" \
  --batch_size "$BSZ" \
  --shard_id "$SHARD_ID" \
  --num_shards "$NUM_SHARDS" \
  $EXTRA \
  2>&1 | tee "${OUT_DIR}/${MODE}_bsz${BSZ}_shard${SHARD_ID}.log"

echo "Done: ${DATASET} ${MODE} shard ${SHARD_ID}"

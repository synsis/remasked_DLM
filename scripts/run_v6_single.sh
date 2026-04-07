#!/usr/bin/env bash
# Run one shard with parameterized lowprob config → results_v6
# Usage: bash run_v6_single.sh <dataset> <shard_id> <num_shards> <mode> <bsz> <tau> <c_max> <rho>
set -euo pipefail
cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask

export https_proxy="100.68.162.212:3128"
export http_proxy="100.68.162.212:3128"
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_ENDPOINT="https://huggingface.co"
export HUGGINGFACE_HUB_ENDPOINT="https://huggingface.co"

DATASET=${1:?Usage: bash run_v6_single.sh <dataset> <shard_id> <num_shards> <mode> <bsz> <tau> <c_max> <rho>}
SHARD_ID=${2:?}
NUM_SHARDS=${3:?}
MODE=${4:?}  # original or remask
BSZ=${5:-1}
TAU=${6:?}
C_MAX=${7:?}
RHO=${8:?}

STRATEGY="low_prob"
TAG="lowprob_t${TAU}_c${C_MAX}_r${RHO}"

OUT_DIR="results_v6/${DATASET}/${TAG}"
mkdir -p "$OUT_DIR"

echo "Dataset: ${DATASET}, Shard: ${SHARD_ID}/${NUM_SHARDS}, Mode: ${MODE}, BSZ: ${BSZ}"
echo "Config: τ=${TAU}, C=${C_MAX}, ρ=${RHO}"
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

echo "Done: ${DATASET} ${MODE} shard ${SHARD_ID} (τ=${TAU}, C=${C_MAX}, ρ=${RHO})"

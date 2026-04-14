#!/usr/bin/env bash
# Run AIME2025 boxed prompt with a specific hyperparam combo
# Usage: bash run_boxed_sweep_single.sh <tau> <c_max> <rho>
set -euo pipefail
cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask

export https_proxy="100.68.162.212:3128"
export http_proxy="100.68.162.212:3128"
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_ENDPOINT="https://huggingface.co"
export HUGGINGFACE_HUB_ENDPOINT="https://huggingface.co"

TAU=${1:?Usage: bash run_boxed_sweep_single.sh <tau> <c_max> <rho>}
C_MAX=${2:?}
RHO=${3:?}
TAG="boxed_t${TAU}_c${C_MAX}_r${RHO}"

OUT_DIR="results_bisect/${TAG}"
mkdir -p "$OUT_DIR"

conda run -n remask python -u -m eval.aime2025 \
  --output_dir "$OUT_DIR" \
  --mode remask \
  --gen_length 16384 \
  --strategy low_prob \
  --remask_threshold "$TAU" \
  --max_remask_per_pos "$C_MAX" \
  --max_remask_ratio "$RHO" \
  2>&1 | tee "${OUT_DIR}/remask.log"

echo "Done: ${TAG}"

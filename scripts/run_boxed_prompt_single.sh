#!/usr/bin/env bash
# Run AIME2025 with standard \boxed{} prompt
# Usage: bash run_boxed_prompt_single.sh <mode>
set -euo pipefail
cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask

export https_proxy="100.68.162.212:3128"
export http_proxy="100.68.162.212:3128"
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_ENDPOINT="https://huggingface.co"
export HUGGINGFACE_HUB_ENDPOINT="https://huggingface.co"

MODE=${1:?Usage: bash run_boxed_prompt_single.sh <mode>}

OUT_DIR="results_bisect/boxed_prompt"
mkdir -p "$OUT_DIR"

EXTRA=""
if [ "$MODE" = "remask" ]; then
  EXTRA="--strategy low_prob --remask_threshold 0.5 --max_remask_per_pos 3 --max_remask_ratio 0.25"
fi

conda run -n remask python -u -m eval.aime2025 \
  --output_dir "$OUT_DIR" \
  --mode "$MODE" \
  --gen_length 16384 \
  $EXTRA \
  2>&1 | tee "${OUT_DIR}/${MODE}.log"

echo "Done: boxed_prompt ${MODE}"

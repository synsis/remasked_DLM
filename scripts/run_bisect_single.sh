#!/usr/bin/env bash
# Bisect AIME2025 regression: run one test configuration
# Usage: bash run_bisect_single.sh <test_id> <mode>
#   test_id: A_old_code | B_lazy_mask | C_head_oldprompt
#   mode:    original | remask
set -euo pipefail

export https_proxy="100.68.162.212:3128"
export http_proxy="100.68.162.212:3128"
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_ENDPOINT="https://huggingface.co"
export HUGGINGFACE_HUB_ENDPOINT="https://huggingface.co"

BASE=/vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask
TEST_ID=${1:?Usage: bash run_bisect_single.sh <test_id> <mode>}
MODE=${2:?}

TAU="0.5"
GEN_LENGTH=16384

OUT_DIR="${BASE}/results_bisect/${TEST_ID}"
mkdir -p "$OUT_DIR"

case "$TEST_ID" in
  A_old_code)
    # Commit 0b3b001: old modeling (pre-computed attn mask + KV cache), old prompt (####)
    # loader defaults: max_remask_per_pos=3, max_remask_ratio=0.25
    CODE_DIR="${BASE}/bisect/code_0b3b001"
    cd "$CODE_DIR"
    EXTRA=""
    if [ "$MODE" = "remask" ]; then
      EXTRA="--strategy low_prob --remask_threshold $TAU"
    fi
    conda run -n remask python -u -m eval.aime2025 \
      --output_dir "$OUT_DIR" \
      --mode "$MODE" \
      --gen_length "$GEN_LENGTH" \
      $EXTRA \
      2>&1 | tee "${OUT_DIR}/${MODE}.log"
    ;;

  B_lazy_mask)
    # Commit 39d7bac: lazy attn mask, new prompt (Answer:), gen_length=16384
    # loader defaults: max_remask_per_pos=3, max_remask_ratio=0.25
    CODE_DIR="${BASE}/bisect/code_39d7bac"
    cd "$CODE_DIR"
    EXTRA=""
    if [ "$MODE" = "remask" ]; then
      EXTRA="--strategy low_prob --remask_threshold $TAU"
    fi
    conda run -n remask python -u -m eval.aime2025 \
      --output_dir "$OUT_DIR" \
      --mode "$MODE" \
      --gen_length "$GEN_LENGTH" \
      $EXTRA \
      2>&1 | tee "${OUT_DIR}/${MODE}.log"
    ;;

  C_head_oldprompt)
    # Current HEAD with old prompt (#### format), gen_length=16384
    cd "$BASE"
    EXTRA=""
    if [ "$MODE" = "remask" ]; then
      EXTRA="--strategy low_prob --remask_threshold $TAU --max_remask_per_pos 3 --max_remask_ratio 0.25"
    fi
    conda run -n remask python -u -m eval.aime2025_oldprompt \
      --output_dir "$OUT_DIR" \
      --mode "$MODE" \
      --gen_length "$GEN_LENGTH" \
      $EXTRA \
      2>&1 | tee "${OUT_DIR}/${MODE}.log"
    ;;

  *)
    echo "Unknown test_id: $TEST_ID"
    exit 1
    ;;
esac

echo "Done: ${TEST_ID} ${MODE}"

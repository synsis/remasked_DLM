#!/usr/bin/env bash
# ── LLaDA2.1 Remask Full Evaluation ──
#
# 用法:
#   bash run_all_evals.sh                                          # 全量
#   bash run_all_evals.sh --max_samples 20                         # 小样本调参
#   BENCHMARKS="gsm_plus mmlu_pro" bash run_all_evals.sh           # 只跑指定
#   STRATEGIES="low_prob" THRESHOLDS="0.05 0.1 0.2" bash run_all_evals.sh

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"

# ── 全部 benchmark (论文 33 个, 按可行性分组) ──
ALL_BENCHMARKS_SIMPLE="gsm_plus cmath aime2025 olympiadbench omni_math mmlu_pro gpqa ceval triviaqa phybench bbh bbh_extra_hard bbh_zh hellaswag piqa prontoqa ocnli kor_bench drop squad2 musr zebralogic cruxeval"
ALL_BENCHMARKS_CODE="humaneval mbpp"
ALL_BENCHMARKS_OFFLINE="ifeval bfcl"

BENCHMARKS="${BENCHMARKS:-$ALL_BENCHMARKS_SIMPLE $ALL_BENCHMARKS_CODE}"
STRATEGIES="${STRATEGIES:-low_prob t2t_remask logit_diff}"
THRESHOLDS="${THRESHOLDS:-}"

MAX=""
EXTRA=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max_samples)  MAX="--max_samples $2"; shift 2 ;;
        --benchmarks)   BENCHMARKS="$2"; shift 2 ;;
        --strategies)   STRATEGIES="$2"; shift 2 ;;
        --thresholds)   THRESHOLDS="$2"; shift 2 ;;
        *)              EXTRA="$EXTRA $1"; shift ;;
    esac
done

echo "=== LLaDA2.1 Remask Eval ==="
echo "benchmarks: $BENCHMARKS"
echo "strategies: $STRATEGIES"
echo "thresholds: ${THRESHOLDS:-<default per strategy>}"
echo ""

run_bench() {
    local B="$1" MODE="$2" STRATEGY="${3:-}" THR_ARG="${4:-}"

    echo -e "\n>>> $B [$MODE ${STRATEGY:+strategy=$STRATEGY }${THR_ARG:+$THR_ARG}]"
    $PYTHON -u -m eval."$B" --mode "$MODE" \
        ${STRATEGY:+--strategy "$STRATEGY"} \
        $THR_ARG \
        --output_dir "results/$B" \
        $MAX $EXTRA || echo "  [WARN] $B failed"

    if [[ "$B" == "humaneval" || "$B" == "mbpp" ]]; then
        local TAG
        if [[ "$MODE" == "original" ]]; then
            TAG="original"
        else
            local T="${THR_ARG##* }"
            TAG="remask_${STRATEGY}_${T:-None}"
        fi
        $PYTHON -m evalplus.evaluate --dataset "${B/humaneval/humaneval}" \
            --samples "results/$B/${TAG}_samples.jsonl" 2>&1 \
            | tee "results/$B/${TAG}_evalplus.log" || true
    fi
}

# ── 1) original baseline ──
for B in $BENCHMARKS; do
    run_bench "$B" original
done

# ── 2) remask: strategy × threshold ──
for S in $STRATEGIES; do
    if [ -z "$THRESHOLDS" ]; then
        THR_LIST="__default__"
    else
        THR_LIST="$THRESHOLDS"
    fi
    for T in $THR_LIST; do
        THR_ARG=""
        [[ "$T" != "__default__" ]] && THR_ARG="--remask_threshold $T"
        for B in $BENCHMARKS; do
            run_bench "$B" remask "$S" "$THR_ARG"
        done
    done
done

# ── 3) Summary table ──
echo -e "\n\n=== Summary ==="
printf "%-20s %-35s %s\n" "Benchmark" "Tag" "Metric"
printf "%-20s %-35s %s\n" "--------" "---" "------"
for B in $BENCHMARKS; do
    for f in results/$B/*_summary.json; do
        [ -f "$f" ] || continue
        TAG=$($PYTHON -c "import json;d=json.load(open('$f'));print(d.get('tag','?'))" 2>/dev/null)
        ACC=$($PYTHON -c "import json;d=json.load(open('$f'));a=d.get('accuracy',d.get('em',d.get('f1','?')));print(f'{a:.4f}' if isinstance(a,float) else a)" 2>/dev/null)
        printf "%-20s %-35s %s\n" "$B" "$TAG" "$ACC"
    done
done

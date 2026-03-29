#!/usr/bin/env bash
# ── 小数据集快速调参 ──
# 在小样本上遍历 remask 策略 × 阈值，找最优配置
#
# 用法:
#   bash run_tune.sh                    # 默认: 20 样本, 5 个快 benchmark
#   bash run_tune.sh --max_samples 50   # 50 样本
#   N=50 bash run_tune.sh               # 等价写法

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/vepfs-mlp2/c20250506/251105017/miniforge3/envs/remask/bin/python}"
N="${N:-20}"

# 调参用的 benchmark（小 + 快 + 多领域覆盖）
TUNE_BENCHMARKS="${TUNE_BENCHMARKS:-gsm_plus mmlu_pro hellaswag piqa cmath}"

# 策略 × 阈值网格
STRATEGIES="${STRATEGIES:-low_prob t2t_remask logit_diff}"
LOW_PROB_THRS="0.05 0.1 0.15 0.2 0.3"
T2T_REMASK_THRS="0.7 0.8 0.9 0.95"
LOGIT_DIFF_THRS="0.1 0.2 0.3 0.5"

EXTRA=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max_samples|-n)  N="$2"; shift 2 ;;
        *)                 EXTRA="$EXTRA $1"; shift ;;
    esac
done

echo "=== Remask Parameter Tuning ==="
echo "samples per benchmark: $N"
echo "benchmarks: $TUNE_BENCHMARKS"
echo ""

# ── 1) Original baseline ──
for B in $TUNE_BENCHMARKS; do
    echo ">>> $B [original] (n=$N)"
    $PYTHON -u -m eval."$B" --mode original --max_samples "$N" \
        --output_dir "results_tune/$B" $EXTRA 2>&1 | tail -3
done

# ── 2) Remask grid search ──
for S in $STRATEGIES; do
    case "$S" in
        low_prob)    THRS="$LOW_PROB_THRS" ;;
        t2t_remask)  THRS="$T2T_REMASK_THRS" ;;
        logit_diff)  THRS="$LOGIT_DIFF_THRS" ;;
    esac

    for T in $THRS; do
        for B in $TUNE_BENCHMARKS; do
            echo ">>> $B [remask: $S thr=$T] (n=$N)"
            $PYTHON -u -m eval."$B" --mode remask --strategy "$S" \
                --remask_threshold "$T" --max_samples "$N" \
                --output_dir "results_tune/$B" $EXTRA 2>&1 | tail -3
        done
    done
done

# ── 3) Results table ──
echo -e "\n\n=============================="
echo "  TUNING RESULTS (n=$N)"
echo "=============================="
printf "%-15s %-30s %s\n" "Benchmark" "Config" "Accuracy"
printf "%-15s %-30s %s\n" "---------" "------" "--------"

for B in $TUNE_BENCHMARKS; do
    for f in results_tune/$B/*_summary.json; do
        [ -f "$f" ] || continue
        TAG=$($PYTHON -c "import json;d=json.load(open('$f'));print(d.get('tag','?'))" 2>/dev/null)
        ACC=$($PYTHON -c "import json;d=json.load(open('$f'));a=d.get('accuracy',d.get('em',d.get('f1','?')));print(f'{a:.4f}' if isinstance(a,float) else a)" 2>/dev/null)
        printf "%-15s %-30s %s\n" "$B" "$TAG" "$ACC"
    done
done | sort -k1,1 -k3,3rn

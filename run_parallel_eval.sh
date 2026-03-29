#!/usr/bin/env bash
# ── Multi-GPU Parallel Evaluation ──
#
# Splits evaluation across multiple GPU groups for N× speedup.
# Each group loads its own model copy and processes a shard of the dataset.
#
# 用法:
#   # Auto-detect GPUs, 2 shards per GPU group of 2 GPUs:
#   bash run_parallel_eval.sh --bench gsm_plus --num_shards 2
#
#   # Specify GPU groups explicitly:
#   bash run_parallel_eval.sh --bench gsm_plus --gpus "0,1" "2,3"
#
#   # With batching inside each shard:
#   bash run_parallel_eval.sh --bench gsm_plus --num_shards 2 --batch_size 4
#
#   # Remask mode:
#   bash run_parallel_eval.sh --bench gsm_plus --num_shards 2 --mode remask --strategy low_prob
#
#   # Multiple benchmarks:
#   bash run_parallel_eval.sh --bench "gsm_plus cmath" --num_shards 2

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"

# ── Defaults ──
NUM_SHARDS=""
BATCH_SIZE=1
MODE="original"
STRATEGY=""
THR_ARG=""
BENCHMARKS=""
MAX_SAMPLES=""
GPU_GROUPS=()
EXTRA_ARGS=""

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bench|--benchmarks)   BENCHMARKS="$2"; shift 2 ;;
        --num_shards)           NUM_SHARDS="$2"; shift 2 ;;
        --batch_size)           BATCH_SIZE="$2"; shift 2 ;;
        --mode)                 MODE="$2"; shift 2 ;;
        --strategy)             STRATEGY="$2"; shift 2 ;;
        --remask_threshold)     THR_ARG="--remask_threshold $2"; shift 2 ;;
        --max_samples)          MAX_SAMPLES="--max_samples $2"; shift 2 ;;
        --gpus)
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                GPU_GROUPS+=("$1")
                shift
            done
            ;;
        *)                      EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

if [ -z "$BENCHMARKS" ]; then
    echo "Usage: $0 --bench <benchmark(s)> --num_shards N [--batch_size B] [--mode original|remask] ..."
    exit 1
fi

# Determine number of shards from GPU groups or --num_shards
if [ ${#GPU_GROUPS[@]} -gt 0 ]; then
    NUM_SHARDS=${#GPU_GROUPS[@]}
elif [ -z "$NUM_SHARDS" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
    NUM_SHARDS=$NUM_GPUS
    echo "Auto-detected $NUM_GPUS GPUs, using $NUM_SHARDS shards (1 GPU each)"
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        GPU_GROUPS+=("$i")
    done
fi

# If GPU groups not set, distribute GPUs evenly
if [ ${#GPU_GROUPS[@]} -eq 0 ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "$NUM_SHARDS")
    GPUS_PER_SHARD=$((NUM_GPUS / NUM_SHARDS))
    if [ "$GPUS_PER_SHARD" -lt 1 ]; then GPUS_PER_SHARD=1; fi
    for s in $(seq 0 $((NUM_SHARDS - 1))); do
        START=$((s * GPUS_PER_SHARD))
        END=$((START + GPUS_PER_SHARD - 1))
        GROUP=""
        for g in $(seq $START $END); do
            [ -n "$GROUP" ] && GROUP="$GROUP,"
            GROUP="$GROUP$g"
        done
        GPU_GROUPS+=("$GROUP")
    done
fi

echo "=== Parallel Eval ==="
echo "benchmarks:  $BENCHMARKS"
echo "num_shards:  $NUM_SHARDS"
echo "batch_size:  $BATCH_SIZE"
echo "mode:        $MODE"
echo "GPU groups:  ${GPU_GROUPS[*]}"
echo ""

STRATEGY_ARG=""
[ -n "$STRATEGY" ] && STRATEGY_ARG="--strategy $STRATEGY"

for BENCH in $BENCHMARKS; do
    echo ">>> Starting $BENCH with $NUM_SHARDS parallel shards..."

    PIDS=()
    for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
        GPU="${GPU_GROUPS[$SHARD]}"
        LOG="results/${BENCH}/.shard_${SHARD}.log"
        mkdir -p "results/${BENCH}"

        echo "  shard $SHARD (GPUs: $GPU) → $LOG"
        CUDA_VISIBLE_DEVICES="$GPU" $PYTHON -u -m eval."$BENCH" \
            --mode "$MODE" \
            $STRATEGY_ARG \
            $THR_ARG \
            --output_dir "results/$BENCH" \
            --batch_size "$BATCH_SIZE" \
            --shard_id "$SHARD" \
            --num_shards "$NUM_SHARDS" \
            $MAX_SAMPLES \
            $EXTRA_ARGS \
            > "$LOG" 2>&1 &
        PIDS+=($!)
    done

    # Wait for all shards
    FAILED=0
    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            echo "  [ERROR] shard $i (PID ${PIDS[$i]}) failed!"
            FAILED=1
        fi
    done

    if [ "$FAILED" -eq 0 ]; then
        echo "  All shards complete. Merging..."
        $PYTHON -c "
import json, glob, sys

bench = '$BENCH'
mode = '$MODE'
strategy = '${STRATEGY}'
thr_arg = '${THR_ARG}'.replace('--remask_threshold ', '')

if mode == 'original':
    tag = 'original'
else:
    tag = f'remask_{strategy}_{thr_arg}' if thr_arg else f'remask_{strategy}_None'

pattern = f'results/{bench}/{tag}_shard*_results.jsonl'
files = sorted(glob.glob(pattern))
if not files:
    print(f'  No shard files found: {pattern}')
    sys.exit(0)

all_results = []
for f in files:
    with open(f) as fh:
        for line in fh:
            if line.strip():
                all_results.append(json.loads(line))

merged_path = f'results/{bench}/{tag}_results.jsonl'
with open(merged_path, 'w') as fh:
    for r in all_results:
        fh.write(json.dumps(r) + '\n')

correct = sum(1 for r in all_results if r.get('correct'))
total = len(all_results)
acc = correct / total if total else 0

summary_path = f'results/{bench}/{tag}_summary.json'
with open(summary_path, 'w') as fh:
    json.dump(dict(
        benchmark=bench, tag=tag, mode=mode,
        accuracy=acc, correct=correct, total=total,
        num_shards=int('$NUM_SHARDS'),
    ), fh, indent=2)

print(f'  Merged {len(files)} shards → {merged_path}')
print(f'  {bench} [{tag}] {correct}/{total} = {acc:.4f}')
"
    fi
    echo ""
done

echo "=== Done ==="

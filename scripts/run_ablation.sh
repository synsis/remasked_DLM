#!/usr/bin/env bash
# Full ablation: Strategy(τ) × C_max × ρ_max on CMATH random 100 samples
# 100 configs total, each run with 8-GPU shard parallel, batch_size=16
set -uo pipefail
cd "$(dirname "$0")/.."

ENV="${CONDA_ENV:-remask}"
BSZ=16
NGPU=8
MAX_SAMPLES=100
SEED=42
OUT_BASE="results_v2/ablation"

export https_proxy="100.68.162.212:3128"
export http_proxy="100.68.162.212:3128"

ALL_GPUS=(0 1 2 3 4 5 6 7)

run_one() {
  local mode=$1 strategy=$2 threshold=$3 c_max=$4 rho=$5
  local tag
  if [ "$mode" = "original" ]; then
    tag="original"
  else
    tag="remask_${strategy}_${threshold}_c${c_max}_r${rho}"
  fi
  local out_dir="${OUT_BASE}/${tag}"

  # Skip if already done
  if [ -f "${out_dir}/summary.json" ]; then
    echo "  SKIP: ${tag} (already done)"
    return
  fi

  mkdir -p "$out_dir"
  echo -n "  ${tag} ... "

  pids=()
  for ((i=0; i<NGPU; i++)); do
    gpu_id=${ALL_GPUS[$i]}
    local extra_args=""
    if [ "$mode" = "remask" ]; then
      extra_args="--strategy $strategy --remask_threshold $threshold --max_remask_per_pos $c_max --max_remask_ratio $rho"
    fi
    CUDA_VISIBLE_DEVICES=$gpu_id PYTHONUNBUFFERED=1 conda run -n $ENV \
      python -u -m eval.cmath \
        --output_dir "$out_dir" \
        --mode "$mode" \
        --batch_size $BSZ \
        --shard_id $i \
        --num_shards $NGPU \
        --max_samples $MAX_SAMPLES \
        --sample_seed $SEED \
        $extra_args \
      > "${out_dir}/shard${i}.log" 2>&1 &
    pids+=($!)
  done
  for ((i=0; i<NGPU; i++)); do
    wait "${pids[$i]}" 2>/dev/null || true
  done

  # Merge
  python3 -c "
import json, os, glob
out_dir = '$out_dir'; tag = '$tag'
results = []
for p in sorted(glob.glob(os.path.join(out_dir, '*_results.jsonl'))):
    with open(p) as f:
        for line in f:
            if line.strip(): results.append(json.loads(line))
if results:
    correct = sum(1 for r in results if r.get('correct'))
    total = len(results)
    acc = correct / total
    print(f'{correct}/{total} = {acc*100:.1f}%')
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump({'tag': tag, 'correct': correct, 'total': total, 'accuracy': acc}, f, indent=2)
else:
    print('NO RESULTS')
"
}

echo "===== Full Ablation: Strategy(τ) × C_max × ρ_max ====="
echo "  CMATH ${MAX_SAMPLES} samples (seed=${SEED}), BSZ=${BSZ}, ${NGPU} GPUs"
echo ""

# Baseline
echo "=== Baseline ==="
run_one original "" "" 3 0.25

# Full cross
CONFIG_NUM=1
for strategy_block in "low_prob:0.3,0.5,0.7,0.9" "t2t_remask:0.5,0.7,0.9" "logit_diff:0.1,0.2,0.3,0.5"; do
  strategy="${strategy_block%%:*}"
  taus="${strategy_block#*:}"
  echo ""
  echo "=== ${strategy} ==="
  IFS=',' read -ra TAU_ARR <<< "$taus"
  for tau in "${TAU_ARR[@]}"; do
    for c_max in 1 3 5; do
      for rho in 0.25 0.50 1.0; do
        run_one remask "$strategy" "$tau" "$c_max" "$rho"
        CONFIG_NUM=$((CONFIG_NUM + 1))
      done
    done
  done
done

echo ""
echo "===== Done: ${CONFIG_NUM} configs ====="
echo ""

# Final summary table
python3 << 'PYEOF'
import json, os, glob

out_base = "results_v2/ablation"
summaries = []
for d in sorted(glob.glob(os.path.join(out_base, "*/summary.json"))):
    with open(d) as f:
        summaries.append(json.load(f))

if not summaries:
    print("No results found!")
    exit()

print(f"{'Tag':55s} {'Correct':>7s} {'Total':>5s} {'Acc%':>6s}")
print("-" * 76)
for s in sorted(summaries, key=lambda x: x['tag']):
    print(f"{s['tag']:55s} {s['correct']:7d} {s['total']:5d} {s['accuracy']*100:6.1f}")

# Save combined
with open(os.path.join(out_base, "all_results.json"), "w") as f:
    json.dump(summaries, f, indent=2)
print(f"\nSaved to {out_base}/all_results.json")
PYEOF

#!/usr/bin/env bash
# Run a single ablation config on 1 GPU.
# Usage: bash run_ablation_single.sh <config_id>
set -euo pipefail
cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask

export https_proxy="100.68.162.212:3128"
export http_proxy="100.68.162.212:3128"
export HF_TOKEN="${HF_TOKEN:-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}"

CONFIG_ID=${1:?Usage: bash run_ablation_single.sh <config_id>}

# Resolve config → tag, mode, args
eval $(python3 /dev/stdin "$CONFIG_ID" << 'PYEOF'
import sys

configs = []
cid = 0
configs.append((cid, "original", "", "", 3, 0.25)); cid += 1
for strategy, taus in [("low_prob", [0.3,0.5,0.7,0.9]),
                        ("t2t_remask", [0.5,0.7,0.9]),
                        ("logit_diff", [0.1,0.2,0.3,0.5])]:
    for tau in taus:
        for c_max in [1, 3, 5]:
            for rho in [0.25, 0.50, 1.0]:
                configs.append((cid, "remask", strategy, str(tau), c_max, rho)); cid += 1

idx = int(sys.argv[1])
c = configs[idx]
if c[1] == "original":
    tag = "original"
    print(f'TAG="{tag}"')
    print(f'MODE="original"')
    print(f'EXTRA_ARGS=""')
else:
    tag = f"{c[2]}_t{c[3]}_c{c[4]}_r{c[5]}"
    print(f'TAG="{tag}"')
    print(f'MODE="remask"')
    print(f'EXTRA_ARGS="--strategy {c[2]} --remask_threshold {c[3]} --max_remask_per_pos {c[4]} --max_remask_ratio {c[5]}"')
PYEOF
)

OUT_DIR="results_v2/ablation/${TAG}"
mkdir -p "$OUT_DIR"

# Skip if already done
if [ -f "${OUT_DIR}/summary.json" ]; then
  echo "SKIP: ${TAG} (already done)"
  exit 0
fi

echo "Running: ${TAG} -> ${OUT_DIR}"

conda run -n remask python -u -m eval.cmath \
  --output_dir "$OUT_DIR" \
  --mode "$MODE" \
  --batch_size 1 \
  --max_samples 100 \
  --sample_seed 42 \
  $EXTRA_ARGS \
  2>&1 | tee "${OUT_DIR}/run.log"

# Create summary
python3 << MERGE
import json, glob, os
out_dir = "$OUT_DIR"
tag = "$TAG"
results = []
for p in sorted(glob.glob(os.path.join(out_dir, "*_results.jsonl"))):
    with open(p) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
if results:
    correct = sum(1 for r in results if r.get("correct"))
    total = len(results)
    acc = correct / total
    print(f"{tag}: {correct}/{total} = {acc*100:.1f}%")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"tag": tag, "correct": correct, "total": total, "accuracy": acc}, f, indent=2)
else:
    print("ERROR: no results!")
MERGE

echo "Done: ${TAG}"

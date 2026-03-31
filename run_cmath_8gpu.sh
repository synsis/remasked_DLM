#!/usr/bin/env bash
# cmath: 8-GPU shard parallel, original then remask low_prob
set -uo pipefail
cd "$(dirname "$0")"

export https_proxy="100.68.162.212:3128"
export http_proxy="100.68.162.212:3128"

ENV="${CONDA_ENV:-remask}"
OUT="results_v2/cmath"
BSZ=64
SHARDS=8

mkdir -p "$OUT"

run_mode() {
  local mode=$1
  local extra_args="${2:-}"
  echo ""
  echo "========== cmath  mode=$mode  bsz=$BSZ  shards=$SHARDS =========="

  pids=()
  for ((i=0; i<SHARDS; i++)); do
    echo "[GPU $i] shard $i/$SHARDS  mode=$mode"
    CUDA_VISIBLE_DEVICES=$i conda run -n $ENV \
      python -m eval.cmath \
        --output_dir "$OUT" \
        --mode "$mode" \
        --batch_size $BSZ \
        --shard_id $i \
        --num_shards $SHARDS \
        $extra_args \
      &
    pids+=($!)
  done

  failed=0
  for ((i=0; i<SHARDS; i++)); do
    wait "${pids[$i]}" 2>/dev/null || true
    ec=$?
    if [ $ec -ne 0 ]; then
      echo "[GPU $i] FAILED exit=$ec"
      failed=$((failed + 1))
    else
      echo "[GPU $i] done"
    fi
  done

  echo "mode=$mode: $((SHARDS - failed))/$SHARDS shards succeeded"

  # merge shard results
  python3 -c "
import json, glob, os
mode = '$mode'
tag = 'original' if mode == 'original' else 'remask_low_prob_None'
out_dir = '$OUT'

all_results = []
for shard in range($SHARDS):
    p = os.path.join(out_dir, f'{tag}_shard{shard}_results.jsonl')
    if not os.path.exists(p):
        continue
    with open(p) as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))

if not all_results:
    print(f'  No results for {tag}')
else:
    # merge into single results + summary
    merged_path = os.path.join(out_dir, f'{tag}_results.jsonl')
    with open(merged_path, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    correct = sum(1 for r in all_results if r.get('correct'))
    total = len(all_results)
    acc = correct / total if total else 0

    # read one shard summary for gen params
    params = {}
    for shard in range($SHARDS):
        sp = os.path.join(out_dir, f'{tag}_shard{shard}_summary.json')
        if os.path.exists(sp):
            params = json.load(open(sp))
            break

    summary = dict(
        benchmark='cmath', tag=tag, mode=mode,
        accuracy=acc, correct=correct, total=total,
        done=True, shards=$SHARDS,
    )
    for k in ('gen_length','block_length','steps','threshold','editing_threshold',
              'temperature','batch_size','strategy','remask_threshold'):
        if k in params:
            summary[k] = params[k]

    # aggregate gen stats
    tpfs = [r.get('_tpf') for r in all_results if r.get('_tpf') is not None]
    tpss = [r.get('_tps') for r in all_results if r.get('_tps') is not None]
    fwds = [r.get('_forward_passes') for r in all_results if r.get('_forward_passes') is not None]
    toks = [r.get('_output_tokens') for r in all_results if r.get('_output_tokens') is not None]
    if tpfs: summary['avg_tpf'] = sum(tpfs)/len(tpfs)
    if tpss: summary['avg_tps'] = sum(tpss)/len(tpss)
    if fwds:
        summary['avg_forward_passes'] = sum(fwds)/len(fwds)
        summary['total_forward_passes'] = sum(fwds)
    if toks:
        summary['avg_output_tokens'] = sum(toks)/len(toks)
        summary['total_output_tokens'] = sum(toks)

    merged_summary = os.path.join(out_dir, f'{tag}_summary.json')
    with open(merged_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  {tag}: {correct}/{total} = {acc:.4f}  (merged from {$SHARDS} shards)')
"
}

run_mode "original"
run_mode "remask" "--strategy low_prob"

echo ""
echo "===== Done ====="

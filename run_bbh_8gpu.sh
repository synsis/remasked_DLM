#!/usr/bin/env bash
# bbh: 8-GPU shard parallel, original then remask low_prob
set -uo pipefail
cd "$(dirname "$0")"

export https_proxy="100.68.162.212:3128"
export http_proxy="100.68.162.212:3128"

ENV="${CONDA_ENV:-remask}"
EVAL="eval.bbh"
OUT="results_v2/bbh"
BSZ=64
SHARDS=8
BENCHMARK="bbh"

mkdir -p "$OUT"

run_mode() {
  local mode=$1
  local extra_args="${2:-}"
  echo ""
  echo "========== $BENCHMARK  mode=$mode  bsz=$BSZ  shards=$SHARDS =========="

  pids=()
  for ((i=0; i<SHARDS; i++)); do
    echo "[GPU $i] shard $i/$SHARDS  mode=$mode"
    CUDA_VISIBLE_DEVICES=$i conda run -n $ENV \
      python -m $EVAL \
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
}

merge() {
  python3 -c "
import json, os

out_dir = '$OUT'
shards = $SHARDS
benchmark = '$BENCHMARK'

for tag, mode in [('original', 'original'), ('remask_low_prob_None', 'remask')]:
    all_results = []
    for s in range(shards):
        p = os.path.join(out_dir, f'{tag}_shard{s}_results.jsonl')
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

    if not all_results:
        print(f'  No results for {tag}')
        continue

    merged_jsonl = os.path.join(out_dir, f'{tag}_results.jsonl')
    with open(merged_jsonl, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    correct = sum(1 for r in all_results if r.get('correct'))
    total = len(all_results)
    acc = correct / total if total else 0

    params = {}
    for s in range(shards):
        sp = os.path.join(out_dir, f'{tag}_shard{s}_summary.json')
        if os.path.exists(sp):
            params = json.load(open(sp))
            break

    summary = dict(
        benchmark=benchmark, tag=tag, mode=mode,
        accuracy=acc, correct=correct, total=total,
        done=True, shards=shards,
    )
    for k in ('gen_length','block_length','steps','threshold','editing_threshold',
              'temperature','batch_size','strategy','remask_threshold'):
        if k in params:
            summary[k] = params[k]

    shard_times = []
    for s in range(shards):
        sp = os.path.join(out_dir, f'{tag}_shard{s}_summary.json')
        if os.path.exists(sp):
            shard_times.append(json.load(open(sp)).get('time_s', 0))
    if shard_times:
        summary['time_s_max_shard'] = max(shard_times)
        summary['time_s_sum_shards'] = sum(shard_times)

    tpfs = [r.get('_tpf') for r in all_results if r.get('_tpf') is not None]
    tpss = [r.get('_tps') for r in all_results if r.get('_tps') is not None]
    fwds = [r.get('_forward_passes') for r in all_results if r.get('_forward_passes') is not None]
    toks = [r.get('_output_tokens') for r in all_results if r.get('_output_tokens') is not None]
    remasks = [r.get('_remask_total') for r in all_results if r.get('_remask_total') is not None]
    if tpfs: summary['avg_tpf'] = sum(tpfs)/len(tpfs)
    if tpss: summary['avg_tps'] = sum(tpss)/len(tpss)
    if fwds:
        summary['avg_forward_passes'] = sum(fwds)/len(fwds)
        summary['total_forward_passes'] = sum(fwds)
    if toks:
        summary['avg_output_tokens'] = sum(toks)/len(toks)
        summary['total_output_tokens'] = sum(toks)
    if remasks:
        summary['avg_remask_total'] = sum(remasks)/len(remasks)
        summary['total_remask'] = sum(remasks)

    merged_summary = os.path.join(out_dir, f'{tag}_summary.json')
    with open(merged_summary, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'{tag}: {correct}/{total} = {acc*100:.2f}%  (wall={summary.get(\"time_s_max_shard\",0):.0f}s)')
"
}

run_mode "original"
run_mode "remask" "--strategy low_prob"

echo ""
echo "===== Merging shards ====="
merge

echo ""
echo "===== Done ====="

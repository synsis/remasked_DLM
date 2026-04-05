#!/usr/bin/env bash
# Resubmit missing shards. Discovers missing shards automatically.
set -euo pipefail
cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask

VOLC=~/.volc/bin/volc
TMP=/tmp/_resub.yml
SUBMITTED=0

submit_one() {
  local DS=$1 MODE=$2 SHARD=$3 NSHARDS=$4
  local OUTDIR="results_v2/best_eval/${DS}/lowprob_t0.3_c1_r0.25"
  local EXTRA=""
  if [ "$MODE" = "remask" ]; then
    EXTRA="--strategy low_prob --remask_threshold 0.3 --max_remask_per_pos 1 --max_remask_ratio 0.25"
  fi
  local TASK_NAME="re-${DS}-${MODE}-s${SHARD}of${NSHARDS}"
  TASK_NAME=$(echo "$TASK_NAME" | tr '.' 'd' | head -c 60)

  cat > "$TMP" << EOF
TaskName: "${TASK_NAME}"
Storages:
  - Type: "Vepfs"
    MountPath: "/vepfs-mlp2/c20250506/251105017"
    SubPath: "/c20250506/251105017"
Entrypoint: "cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask && export HF_TOKEN=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 && bash scripts/run_best_eval_single.sh ${DS} ${SHARD} ${NSHARDS} ${MODE} 1"
Tags: [LLM, yl, llada, remask, best-eval, retry]
ImageUrl: "cr-mlp-cn-beijing.cr.volces.com/public/yl_dllm:1.0"
Framework: "Custom"
ResourceQueueName: "c20250506"
TaskRoleSpecs:
  - RoleName: "worker"
    RoleReplicas: 1
    Flavor: "ml.pni2.3xlarge"
ActiveDeadlineSeconds: 10800
DelayExitTimeSeconds: 0
AccessType: "Private"
Preemptible: true
Priority: 6
EOF

  echo -n "  ${TASK_NAME} ... "
  $VOLC ml_task submit -c "$TMP" 2>&1 | grep -oE "task_id=[^ ]*" || echo "ok"
  sleep 0.2
  SUBMITTED=$((SUBMITTED + 1))
}

# Python finds missing shards
while IFS=' ' read -r DS MODE SHARD NSHARDS; do
  submit_one "$DS" "$MODE" "$SHARD" "$NSHARDS"
done < <(python3 << 'PYEOF'
import os, glob, re, math

base = "results_v2/best_eval"
datasets_info = {
    "humaneval": 164, "ifeval": 541, "aime2025": 30, "gpqa": 198,
    "triviaqa": 17944, "drop": 9535, "hellaswag": 10042, "piqa": 1838,
    "cmath": 1098, "bbh": 6511, "gsm_plus": 2400, "mmlu_pro": 12032,
}

for ds, total in datasets_info.items():
    expected = math.ceil(total / 64)
    d = f"{base}/{ds}/lowprob_t0.3_c1_r0.25"
    orig_shards = set()
    remask_shards = set()
    if os.path.isdir(d):
        for f in glob.glob(f"{d}/original*shard*_summary.json"):
            m = re.search(r'shard(\d+)_summary', os.path.basename(f))
            if m: orig_shards.add(int(m.group(1)))
        for f in glob.glob(f"{d}/remask_*shard*_summary.json"):
            m = re.search(r'shard(\d+)_summary', os.path.basename(f))
            if m: remask_shards.add(int(m.group(1)))
    for s in range(expected):
        if s not in orig_shards:
            print(f"{ds} original {s} {expected}")
        if s not in remask_shards:
            print(f"{ds} remask {s} {expected}")
PYEOF
)

rm -f "$TMP"
echo ""
echo "Total submitted: ${SUBMITTED}"

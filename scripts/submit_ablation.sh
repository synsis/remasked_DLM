#!/usr/bin/env bash
# Submit 100 ablation jobs, task name = hyperparam tag
set -euo pipefail
cd "$(dirname "$0")/.."

VOLC=~/.volc/bin/volc
TMP_YML="/tmp/_abl.yml"
rm -rf results_v2/ablation
mkdir -p results_v2/ablation

# Generate tag for each config
get_tag() {
  python3 -c "
configs = []
cid = 0
configs.append((cid, 'original', '', '', 3, 0.25)); cid += 1
for s, ts in [('low_prob',[0.1,0.3,0.5,0.7,0.9]),('t2t_remask',[0.5,0.7,0.9]),('logit_diff',[0.1,0.2,0.3,0.5])]:
    for t in ts:
        for c in [1,3,5]:
            for r in [0.25,0.50,1.0]:
                configs.append((cid,'remask',s,str(t),c,r)); cid += 1
c = configs[$1]
tag = f'{c[0]:03d}_original' if c[1]=='original' else f'{c[0]:03d}_{c[2]}_t{c[3]}_c{c[4]}_r{c[5]}'
print(tag)
"
}

for ((i=0; i<109; i++)); do
  TAG=$(get_tag $i)

  # Skip if already done
  if [ -f "results_v2/ablation/${TAG}/summary.json" ]; then
    echo "[$i] SKIP ${TAG}"
    continue
  fi

  TASK_NAME="abl-${TAG}"
  # volcengine task name: replace dots/underscores for safety
  TASK_NAME=$(echo "$TASK_NAME" | tr '.' 'd' | head -c 60)

  cat > "$TMP_YML" << EOF
TaskName: "${TASK_NAME}"
Storages:
  - Type: "Vepfs"
    MountPath: "/vepfs-mlp2/c20250506/251105017"
    SubPath: "/c20250506/251105017"
Entrypoint: "cd /vepfs-mlp2/c20250506/251105017/yaolin/LLADA_pretraining/new_llada2.1_infer_remask && export HF_TOKEN=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 && bash scripts/run_ablation_single.sh ${i}"
Tags: [LLM, yl, llada, remask, ablation]
ImageUrl: "cr-mlp-cn-beijing.cr.volces.com/public/yl_dllm:1.0"
Framework: "Custom"
ResourceQueueName: "c20250506"
TaskRoleSpecs:
  - RoleName: "worker"
    RoleReplicas: 1
    Flavor: "ml.pni2.3xlarge"
ActiveDeadlineSeconds: 7200
DelayExitTimeSeconds: 0
AccessType: "Private"
Preemptible: true
Priority: 6
EOF

  echo -n "[$i] ${TASK_NAME} ... "
  $VOLC ml_task submit -c "$TMP_YML" 2>&1 | grep -oE "TaskId[^,]*" || echo "ok"
  sleep 0.3
done

rm -f "$TMP_YML"
echo ""
echo "Done."

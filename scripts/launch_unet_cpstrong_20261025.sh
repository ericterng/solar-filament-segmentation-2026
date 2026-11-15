#!/bin/bash
# Probe 23 (pre-registered 2026-10-25): stronger copy-paste (1-3 paste tries on every crop) on top of
# probe 20's copy-paste U-Net, same remote root / env / cache / recipe as unet_last3_20261024, on ONE
# RTX A5000 chosen at start time after a fresh check and passed by UUID.
#   bash scripts/launch_unet_cpstrong_20261025.sh GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
# Order: smoke (+2-image val logits), then seed 0, then seed 1, each followed by dihedral val and test
# logits. The saved copypaste_s0 / copypaste_s1 runs are the control and are not retrained.
# Stops at the first failure; nothing is retried; never overwrites (run directories must not exist).
set -euo pipefail
GPU_UUID=${1:?usage: launch_unet_cpstrong_20261025.sh <GPU UUID>}
[[ $GPU_UUID =~ ^GPU-[0-9a-f-]{36}$ ]] || { echo "bad GPU UUID: $GPU_UUID" >&2; exit 64; }
cd "$(dirname "$0")/.."
TAG=unet_cpstrong_20261025
exec 9>"$TAG.lock"
flock -n 9 || exit 73
OUT=outputs/unet_cpstrong_20261025
for run in smoke_cpstrong cpstrong_s0 cpstrong_s1; do
    [ ! -e "$OUT/$run" ] || { echo "$OUT/$run exists; refusing to overwrite" >&2; exit 74; }
done
[ ! -e "${TAG}_exit_code" ] || { echo "${TAG}_exit_code exists; move old run files first" >&2; exit 74; }
MON_PID=
trap 'code=$?; [ -n "$MON_PID" ] && kill "$MON_PID" 2>/dev/null; echo "$code" > "${TAG}_exit_code"; date -Is > "${TAG}_finished_at"' EXIT

# The chosen card must exist and carry no compute process / no memory in use right now.
GPUS=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
grep -q ", $GPU_UUID\$" <<< "$GPUS" || { echo "UUID not on this host" >&2; exit 65; }
if grep -q "^$GPU_UUID" <<< "$APPS"; then echo "GPU $GPU_UUID is busy; choose again" >&2; exit 66; fi
used=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "$used" -le 100 ] || { echo "GPU $GPU_UUID has ${used} MiB in use" >&2; exit 66; }

export CUDA_VISIBLE_DEVICES=$GPU_UUID UNET_ENCODER_GPU_UUID=$GPU_UUID
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME=$PWD/hf_home HF_HUB_OFFLINE=1
export FILAMENT_DATA_ROOT=$PWD/data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026
export FILAMENT_OUTPUT_ROOT=$PWD/$OUT
PY=env/bin/python
SPLIT=outputs/amey_unet_fold1_20261010/splits.json
CACHE=outputs/amey_unet_cache
TRAIN_LIMIT=3h   # per training run; probe 20 copy-paste took about 45 min, a hang is a failure
LOGIT_LIMIT=60m  # per logits job; B4 val+test took 12.5 min together
stage() { echo "$1" > "${TAG}_stage"; { date -Is; echo "$1"; } >> "${TAG}_stages.log"; }

mkdir -p "$OUT"
date -Is > "${TAG}_started_at"
echo "$GPU_UUID" > "$OUT/gpu_uuid_${TAG}.txt"
# Process->GPU log every 60 s, so the run can be audited after the fact.
( while true; do { date -Is; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader; } >> "$OUT/gpu_monitor_${TAG}.log"; sleep 60; done ) &
MON_PID=$!

logits() {  # run dir, subset, extra args
    timeout "$LOGIT_LIMIT" $PY third_party/ameyypawar-filament-seg/scripts/cache_logits.py \
        --checkpoint "$1/model_last.pt" --subset "$2" --tta dihedral --split "$SPLIT" \
        --cache-dir "$CACHE" --out-dir "$1/logits_$2" ${3:-} > "$1/logits_$2.log" 2>&1
}
train() {  # seed, run dir, extra args
    timeout "$TRAIN_LIMIT" $PY scripts/train_unet_cpstrong_20261025.py --seed "$1" \
        --out "$2" ${3:-} > "$2.log" 2>&1
}

stage smoke_cpstrong
train 0 "$OUT/smoke_cpstrong" --smoke
logits "$OUT/smoke_cpstrong" val "--limit 2"

for seed in 0 1; do
    run=cpstrong_s$seed
    stage "$run"
    train "$seed" "$OUT/$run"
    stage "${run}_logits"
    logits "$OUT/$run" val
    logits "$OUT/$run" test
done
stage complete

#!/bin/bash
# Probe 28 (docs/experiments/instance_refiner_20261031.md): everything the A5000 does, on ONE RTX A5000
# chosen at start time after a fresh check and passed by UUID.
#   bash scripts/launch_refiner_20261031.sh GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
# Inputs (uploaded from the local step 0): outputs/refiner_20261031/{samples_s0,samples_s1}.pkl,
# {instances_val_s0,instances_val_s1,instances_test}.json, oracle.json (gate_pass must be true).
# Order: smoke (two 40-sample refiners, apply on 3 images, identity on 3, real-data equivalence on 2 stems)
# -> 8 val refiners (s in 0 1 x train half A B x refiner seed 0 1) -> 2 test refiners (s0, all 141, r0 r1)
# -> apply val (4 cells + identity per U-Net seed) -> apply test (r0 + identity) -> verification.json.
# Stops at the first failure; nothing is retried; never overwrites (run directories must not exist).
set -euo pipefail
GPU_UUID=${1:?usage: launch_refiner_20261031.sh <GPU UUID>}
[[ $GPU_UUID =~ ^GPU-[0-9a-f-]{36}$ ]] || { echo "bad GPU UUID: $GPU_UUID" >&2; exit 64; }
cd "$(dirname "$0")/.."
TAG=refiner_20261031
exec 9>"$TAG.lock"
flock -n 9 || exit 73
OUT=outputs/refiner_20261031
CP=outputs/unet_cpstrong_20261025
CACHE=outputs/amey_unet_cache
PY=env/bin/python
for f in samples_s0.pkl samples_s1.pkl instances_val_s0.json instances_val_s1.json instances_test.json oracle.json; do
    [ -e "$OUT/$f" ] || { echo "missing input $OUT/$f" >&2; exit 65; }
done
$PY - <<'EOF' || { echo "oracle gate or sample floor not passed; refusing to train" >&2; exit 65; }
import json, pickle
assert json.load(open('outputs/refiner_20261031/oracle.json'))['gate_pass']
for s in (0, 1):
    assert pickle.load(open(f'outputs/refiner_20261031/samples_s{s}.pkl', 'rb'))['meta']['floor_pass']
EOF
for d in smoke_trainA smoke_trainB r_s0_trainA_r0 r_s0_trainA_r1 r_s0_trainB_r0 r_s0_trainB_r1 \
         r_s1_trainA_r0 r_s1_trainA_r1 r_s1_trainB_r0 r_s1_trainB_r1 r_s0_trainALL_r0 r_s0_trainALL_r1; do
    [ ! -e "$OUT/$d" ] || { echo "$OUT/$d exists; refusing to overwrite" >&2; exit 74; }
done
for f in smoke_apply.json smoke_identity.json smoke_equivalence.json refined_val_s0_r0.json refined_val_s0_r1.json \
         refined_val_s1_r0.json refined_val_s1_r1.json refined_val_s0_identity.json refined_val_s1_identity.json \
         refined_test_r0.json refined_test_identity.json verification.json; do
    [ ! -e "$OUT/$f" ] || { echo "$OUT/$f exists; refusing to overwrite" >&2; exit 74; }
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

export CUDA_VISIBLE_DEVICES=$GPU_UUID
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME=$PWD/hf_home HF_HUB_OFFLINE=1
export FILAMENT_DATA_ROOT=$PWD/data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026
TRAIN_LIMIT=30m   # per refiner; ~1000 steps at batch 8 is a few minutes, a hang is a failure
APPLY_LIMIT=30m   # per apply job
SMOKE_LIMIT=10m
stage() { echo "$1" > "${TAG}_stage"; { date -Is; echo "$1"; } >> "${TAG}_stages.log"; }

date -Is > "${TAG}_started_at"
echo "$GPU_UUID" > "$OUT/gpu_uuid_${TAG}.txt"
( while true; do { date -Is; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader; } >> "$OUT/gpu_monitor_${TAG}.log"; sleep 60; done ) &
MON_PID=$!

train() {  # unet seed, train half (A|B|all), refiner seed, out dir, extra
    timeout "$TRAIN_LIMIT" $PY scripts/train_refiner_20261031.py --samples "$OUT/samples_s$1.pkl" --train-half "$2" \
        --refiner-seed "$3" --init-ckpt "$CP/cpstrong_s$1/model_last.pt" --cache "$CACHE" --out "$OUT/$4" ${5:-} \
        > "$OUT/$4.log" 2>&1
}
apply_val() {  # unet seed, refiner seed
    timeout "$APPLY_LIMIT" $PY scripts/apply_refiner_20261031.py --instances "$OUT/instances_val_s$1.json" --cache "$CACHE" \
        --apply-a "$OUT/r_s$1_trainB_r$2/model_last.pt" --apply-b "$OUT/r_s$1_trainA_r$2/model_last.pt" \
        --out "$OUT/refined_val_s$1_r$2.json" > "$OUT/refined_val_s$1_r$2.log" 2>&1
}

stage smoke
timeout "$SMOKE_LIMIT" $PY scripts/train_refiner_20261031.py --samples "$OUT/samples_s0.pkl" --train-half A --refiner-seed 0 \
    --init-ckpt "$CP/cpstrong_s0/model_last.pt" --cache "$CACHE" --out "$OUT/smoke_trainA" --smoke > "$OUT/smoke_trainA.log" 2>&1
timeout "$SMOKE_LIMIT" $PY scripts/train_refiner_20261031.py --samples "$OUT/samples_s0.pkl" --train-half B --refiner-seed 0 \
    --init-ckpt "$CP/cpstrong_s0/model_last.pt" --cache "$CACHE" --out "$OUT/smoke_trainB" --smoke > "$OUT/smoke_trainB.log" 2>&1
timeout "$SMOKE_LIMIT" $PY scripts/apply_refiner_20261031.py --instances "$OUT/instances_val_s0.json" --cache "$CACHE" --limit 3 \
    --apply-a "$OUT/smoke_trainB/model_last.pt" --apply-b "$OUT/smoke_trainA/model_last.pt" \
    --out "$OUT/smoke_apply.json" > "$OUT/smoke_apply.log" 2>&1
timeout "$SMOKE_LIMIT" $PY scripts/apply_refiner_20261031.py --instances "$OUT/instances_val_s0.json" --cache "$CACHE" --limit 3 \
    --identity --out "$OUT/smoke_identity.json" > "$OUT/smoke_identity.log" 2>&1
STEMS=$($PY -c "import json; print(' '.join(im['stem'] for im in json.load(open('$OUT/instances_val_s0.json'))['images'][:2]))")
timeout "$SMOKE_LIMIT" $PY scripts/apply_refiner_20261031.py --cache "$CACHE" --init-ckpt "$CP/cpstrong_s0/model_last.pt" \
    --logits-dir "$CP/cpstrong_s0/logits_val" --equivalence-check $STEMS --out "$OUT/smoke_equivalence.json" \
    > "$OUT/smoke_equivalence.log" 2>&1
grep -q '"passed": true' "$OUT/smoke_equivalence.json" || { echo "real-data equivalence failed" >&2; exit 67; }

for s in 0 1; do
    for half in A B; do
        for r in 0 1; do
            stage "train r_s${s}_train${half}_r${r}"
            train "$s" "$half" "$r" "r_s${s}_train${half}_r${r}"
        done
    done
done
for r in 0 1; do
    stage "train r_s0_trainALL_r${r}"
    train 0 all "$r" "r_s0_trainALL_r${r}"
done
for s in 0 1; do
    for r in 0 1; do
        stage "apply val s${s} r${r}"
        apply_val "$s" "$r"
    done
    stage "apply val s${s} identity"
    timeout "$APPLY_LIMIT" $PY scripts/apply_refiner_20261031.py --instances "$OUT/instances_val_s$s.json" --cache "$CACHE" \
        --identity --out "$OUT/refined_val_s${s}_identity.json" > "$OUT/refined_val_s${s}_identity.log" 2>&1
done
stage "apply test r0"
timeout "$APPLY_LIMIT" $PY scripts/apply_refiner_20261031.py --instances "$OUT/instances_test.json" --cache "$CACHE" \
    --apply-all "$OUT/r_s0_trainALL_r0/model_last.pt" --out "$OUT/refined_test_r0.json" > "$OUT/refined_test_r0.log" 2>&1
timeout "$APPLY_LIMIT" $PY scripts/apply_refiner_20261031.py --instances "$OUT/instances_test.json" --cache "$CACHE" \
    --identity --out "$OUT/refined_test_identity.json" > "$OUT/refined_test_identity.log" 2>&1

stage verification
$PY - <<'EOF'
import hashlib, json, glob, os
out = 'outputs/refiner_20261031'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
runs = {os.path.basename(d): json.load(open(f'{d}/runtime.json')) for d in sorted(glob.glob(f'{out}/r_s*')) + sorted(glob.glob(f'{out}/smoke_train*')) if os.path.isdir(d)}
files = {os.path.basename(p): sha(p) for p in sorted(glob.glob(f'{out}/refined_*.json')) + sorted(glob.glob(f'{out}/smoke_*.json'))}
checks = {'runs': {k: {kk: v[kk] for kk in ('train_half', 'apply_half', 'unet_seed', 'refiner_seed', 'samples', 'forward_calls',
                                              'epoch_loss', 'checkpoint_sha256', 'equivalence', 'peak_mem_gb', 'seconds', 'smoke')}
                   for k, v in runs.items()},
          'files_sha256': files, 'gpu_uuid': open(f'{out}/gpu_uuid_refiner_20261031.txt').read().strip(),
          'scripts_sha256': {f: sha(f'scripts/{f}') for f in ('refiner_core_20261031.py', 'train_refiner_20261031.py',
                                                                 'apply_refiner_20261031.py', 'launch_refiner_20261031.sh')},
          'inputs_sha256': {f: sha(f'{out}/{f}') for f in ('samples_s0.pkl', 'samples_s1.pkl', 'instances_val_s0.json',
                                                            'instances_val_s1.json', 'instances_test.json', 'oracle.json')}}
formal = [k for k in runs if not runs[k]['smoke']]
assert len(formal) == 10 and all(not runs[k]['smoke'] and runs[k]['epochs'] == 16 for k in formal), formal
json.dump(checks, open(f'{out}/verification.json', 'w'), indent=2)
print('VERIFIED', len(formal), 'refiners', len(files), 'json outputs')
EOF
stage done

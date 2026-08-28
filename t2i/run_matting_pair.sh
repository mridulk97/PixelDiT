#!/usr/bin/env bash
#
# Launch two matting runs side by side, one per GPU, each in its own detached
# screen session.
#
#   bash run_matting_pair.sh d646 both      # baseline vs refinement head
#   bash run_matting_pair.sh am2k both
#
# Why a wrapper rather than two shells:
#
#  * Data setup runs ONCE here. Both arms would otherwise call
#    setup_*_data.sh concurrently and race on the same verify-and-touch pass.
#  * Each arm is pinned to its own GPU. Without that both land on cuda:0 and
#    fight for memory.
#  * `screen -dmS` starts each run already detached, so nothing is ever wired
#    to a terminal. An attached screen is how the earlier runs died: VS Code
#    injects keystrokes into restored terminals, and those reach whatever is
#    running inside screen -- a stray Ctrl+C is signal 2 to the trainer.
#
# Arms differ by exactly one variable so the comparison attributes. Override
# the pair with MATTING_ARM_A / MATTING_ARM_B (name and args, colon-separated).
#
# Environment:
#   MATTING_STEPS       max optimizer steps           (default 3000)
#   MATTING_GPUS        comma-separated GPU ids       (default 0,1)
#   MATTING_RUN_ROOT    run directory root            (default /scratch/mridul/runs/matting/v2)
#   MATTING_BAND_RADIUS "min,max" band radius override (default: config value)
#   MATTING_ARM_A/B     "<suffix>:<extra args>"

set -euo pipefail

dataset="${1:?Usage: bash run_matting_pair.sh <am2k|d646> <patch|pixel|both|sequence|sequence_pixel> [shared extra args...]}"
mode="${2:?Usage: bash run_matting_pair.sh <dataset> <mode> [shared extra args...]}"
shift 2
shared_args=("$@")

case "$dataset" in
  am2k|d646) ;;
  *) echo "Unsupported dataset: $dataset (expected am2k or d646)" >&2; exit 2 ;;
esac

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
steps="${MATTING_STEPS:-3000}"
run_root="${MATTING_RUN_ROOT:-/scratch/mridul/runs/matting/v2}"
stamp="$(date +%Y%m%d_%H%M%S)"

# The two arms. Baseline first so it takes the lower GPU id.
arm_a="${MATTING_ARM_A:-base:}"
arm_b="${MATTING_ARM_B:-head:--model.use_refine_head=true}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the pixel_dit conda environment before launching." >&2
  exit 2
fi
python_bin="$CONDA_PREFIX/bin/python"

IFS=',' read -r -a gpus <<< "${MATTING_GPUS:-0,1}"
visible="$("$python_bin" -c 'import torch; print(torch.cuda.device_count())')"
if (( visible < ${#gpus[@]} )); then
  echo "Requested ${#gpus[@]} GPUs (${MATTING_GPUS:-0,1}) but torch sees $visible." >&2
  echo "Set MATTING_GPUS=0 and run the arms one after another instead." >&2
  exit 2
fi

# The band loss concentrates gradient on the trimap unknown band. Measured band
# coverage at radius 40: AM-2K 22% of pixels (median), D-646 34% and up to 81%,
# because its foregrounds are genuinely transparent. Past roughly a third of the
# frame the term stops being targeted and drifts toward whole-image MSE, so a
# smaller radius is worth trying on D-646.
band_args=()
if [[ -n "${MATTING_BAND_RADIUS:-}" ]]; then
  IFS=',' read -r band_min band_max <<< "$MATTING_BAND_RADIUS"
  band_args=(
    "--train.matting_band_radius_min=${band_min}"
    "--train.matting_band_radius_max=${band_max}"
  )
fi

# Run the dataset setup once, here, rather than letting both arms race on it.
echo "== data =="
bash "$here/$( [[ "$dataset" == "d646" ]] && echo setup_d646_data.sh || echo setup_am2k_data.sh )"
echo

launch() {
  local spec="$1" gpu="$2"
  local suffix="${spec%%:*}"
  local extra="${spec#*:}"
  local name="${dataset}_${mode}_${suffix}_${stamp}"
  local session="m_${dataset}_${suffix}"
  local work_dir="${run_root}/${name}"

  mkdir -p "$work_dir"
  local log="${work_dir}/launch.log"

  # shellcheck disable=SC2086
  # `bash -c`, not `-lc`: carry this shell's environment in explicitly rather
  # than re-running .bashrc and hoping the right env comes back.
  screen -dmS "$session" bash -c "
    export CONDA_PREFIX='${CONDA_PREFIX}'
    export PATH='${CONDA_PREFIX}/bin:${PATH}'
    export PYTHONNOUSERSITE=1
    export CUDA_VISIBLE_DEVICES='${gpu}'
    export MATTING_DATASET='${dataset}'
    export MATTING_RUN_NAME='${name}'
    export MATTING_RUN_ROOT='${run_root}'
    export MATTING_SKIP_DATA_SETUP=1
    cd '${here}'
    bash run_matting_overfit_2gpu.sh '${mode}' \
      --train.max_train_steps=${steps} \
      ${band_args[*]:-} ${shared_args[*]:-} ${extra} \
      2>&1 | tee -a '${log}'
  "
  printf '  %-28s GPU %-3s screen %-22s %s\n' "$name" "$gpu" "$session" "$log"
}

echo "== launching =="
launch "$arm_a" "${gpus[0]}"
launch "$arm_b" "${gpus[1]}"

cat <<EOF

Both arms are detached. Nothing is attached to this terminal, so closing it
(or a VS Code reconnect) cannot signal them.

  screen -ls                       list sessions
  screen -r m_${dataset}_base      attach   (detach again with Ctrl+A then D)
  tail -f ${run_root}/${dataset}_${mode}_base_${stamp}/launch.log

  screen -S m_${dataset}_base -X quit    stop an arm

When they finish, report on both:

  python matting_report.py --dataset ${dataset} \\
    --config configs/PixelDiT_1024px_matting_${dataset}_overfit.yaml \\
    --adapter_path ${run_root}/${dataset}_${mode}_base_${stamp}/adapters/latest.pth \\
    --output_dir  ${run_root}/${dataset}_${mode}_base_${stamp}/report
EOF

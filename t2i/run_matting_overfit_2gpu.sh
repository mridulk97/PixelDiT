#!/usr/bin/env bash
set -euo pipefail

mode="${1:?Usage: bash run_matting_overfit_2gpu.sh patch|pixel|both|sequence|sequence_pixel [extra pyrallis arguments]}"
shift
case "$mode" in
  patch|pixel|both|sequence|sequence_pixel) ;;
  *)
    echo "Unsupported conditioning mode: $mode" >&2
    exit 2
    ;;
esac

# Which dataset this run trains on. AM-2K is animal photographs with hard-ish
# edges; D-646 is composited foregrounds (glass, water, veils, fine hair) and is
# where partial coverage actually lives.
dataset="${MATTING_DATASET:-am2k}"
case "$dataset" in
  am2k) config="configs/PixelDiT_1024px_matting_am2k_overfit.yaml"; data_setup="setup_am2k_data.sh" ;;
  d646) config="configs/PixelDiT_1024px_matting_d646_overfit.yaml"; data_setup="setup_d646_data.sh" ;;
  *)
    echo "Unsupported MATTING_DATASET: $dataset (expected am2k or d646)" >&2
    exit 2
    ;;
esac
config="${MATTING_CONFIG:-$config}"
num_processes="${NPROC_PER_NODE:-1}"
timestamp="${MATTING_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
run_name="${MATTING_RUN_NAME:-pixeldit-matting-${dataset}-${mode}_${timestamp}}"
run_root="${MATTING_RUN_ROOT:-/scratch/mridul/runs/matting/v2}"
work_dir="${MATTING_RUN_DIR:-${run_root}/${run_name}}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the pixel_dit conda environment before launching." >&2
  exit 2
fi
python_bin="$CONDA_PREFIX/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "Conda Python is not executable: $python_bin" >&2
  exit 2
fi

# Do not let ~/.local/bin/torchrun or ~/.local Python packages override the
# active Conda environment. Launch torch.distributed with this interpreter.
export PYTHONNOUSERSITE=1
nvidia_library_path="$("$python_bin" -c '
from pathlib import Path
import sysconfig

site_packages = Path(sysconfig.get_paths()["purelib"])
library_dirs = sorted(str(path) for path in (site_packages / "nvidia").glob("*/lib") if path.is_dir())
torch_lib = site_packages / "torch" / "lib"
if torch_lib.is_dir():
    library_dirs.append(str(torch_lib))
print(":".join(library_dirs))
')"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${nvidia_library_path:+:$nvidia_library_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="${HF_HOME:-/projects/ml4science/HF_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/projects/ml4science/HF_CACHE/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/projects/ml4science/HF_CACHE/datasets}"
export DIFFUSERS_CACHE="${DIFFUSERS_CACHE:-/projects/ml4science/HF_CACHE/diffusers}"

runtime_info="$("$python_bin" -c 'import sys, torch, transformers, wandb; print(f"{sys.executable} | torch={torch.__version__} | transformers={transformers.__version__} | wandb={wandb.__version__}")')"
"$python_bin" -c 'import ctypes; ctypes.CDLL("libnvrtc.so.12")'
visible_gpus="$("$python_bin" -c 'import torch; print(torch.cuda.device_count())')"
if (( visible_gpus < num_processes )); then
  echo "Requested $num_processes processes, but Conda PyTorch sees only $visible_gpus CUDA GPU(s)." >&2
  exit 2
fi

# These datasets live on scratch and get reaped periodically: extraction stamps
# every image with the archive's own build date, so an age-based cleanup treats
# a fresh extract as years stale. Rebuild and re-stamp before every launch.
# Costs about ten seconds when the data is already in place.
if [[ "${MATTING_SKIP_DATA_SETUP:-0}" != "1" ]]; then
  bash "$(dirname "${BASH_SOURCE[0]}")/${data_setup}"
fi

mkdir -p "$work_dir"
export WANDB_DIR="$work_dir"
export WANDB_CACHE_DIR="$work_dir/.wandb_cache"
export WANDB_DATA_DIR="$work_dir/.wandb_data"
export WANDB_ARTIFACT_DIR="$work_dir/artifacts"

echo "PixelDiT matting run: $run_name"
echo "Dataset:              $dataset"
echo "Config:               $config"
echo "Conditioning mode:    $mode"
echo "Run directory:        $work_dir"
echo "Runtime:              $runtime_info"
echo "Visible GPUs:         $visible_gpus (using $num_processes)"

launcher_args=(--nproc_per_node="$num_processes")
if [[ -n "${MASTER_PORT:-}" ]]; then
  # An explicit port remains useful for managed/multi-node launches.
  launcher_args+=(--master_port="$MASTER_PORT")
  echo "Rendezvous:           explicit port $MASTER_PORT"
else
  # torchrun's standalone rendezvous obtains a free local port. This permits
  # independent jobs in separate interactive sessions on the same node.
  launcher_args+=(--standalone)
  echo "Rendezvous:           standalone (automatic free port)"
fi

exec "$python_bin" -m torch.distributed.run \
  "${launcher_args[@]}" \
  train_matting.py \
  --config_path="$config" \
  --model.conditioning_mode="$mode" \
  --work_dir="$work_dir" \
  --name="$run_name" \
  "$@"

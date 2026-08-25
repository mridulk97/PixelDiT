#!/usr/bin/env bash
#SBATCH --job-name=pixeldit_matting
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --partition=h200_normal_q
#SBATCH --account=imageomicswithanuj
#SBATCH --qos=tc_h200_normal_short
#SBATCH --output=/scratch/mridul/runs/matting/v1/slurm-%j-%x.out

set -euo pipefail

mode="${1:-${MATTING_MODE:-both}}"
if (( $# > 0 )); then
  shift
fi
case "$mode" in
  patch|pixel|both|sequence) ;;
  *)
    echo "Unsupported conditioning mode: $mode" >&2
    exit 2
    ;;
esac

module reset
module load Miniconda3/24.7.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pixel_dit

cd /home/mridul/matting/PixelDiT/t2i

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NPROC_PER_NODE=1
unset MASTER_PORT

batch_size_per_gpu="${BATCH_SIZE_PER_GPU:-4}"
gradient_accumulation_steps="${GRAD_ACCUM_STEPS:-2}"
max_train_steps="${MAX_TRAIN_STEPS:-1000}"
timestamp="$(date +%Y%m%d_%H%M%S)"
export MATTING_RUN_NAME="${MATTING_RUN_NAME:-pixeldit-matting-am2k-${mode}-slurm${SLURM_JOB_ID}_${timestamp}}"

echo "Slurm job:             ${SLURM_JOB_ID}"
echo "Mode:                  ${mode}"
echo "Batch/GPU:             ${batch_size_per_gpu}"
echo "Gradient accumulation: ${gradient_accumulation_steps}"
echo "Effective batch:       $((batch_size_per_gpu * gradient_accumulation_steps))"
echo "Maximum steps:         ${max_train_steps}"

exec bash run_matting_overfit_2gpu.sh "$mode" \
  --train.train_batch_size="$batch_size_per_gpu" \
  --train.gradient_accumulation_steps="$gradient_accumulation_steps" \
  --train.max_train_steps="$max_train_steps" \
  "$@"

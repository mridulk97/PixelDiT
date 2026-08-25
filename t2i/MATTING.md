# PixelDiT AM-2K matting pilot

This pilot fine-tunes PixelDiT to generate a three-channel copy of an alpha matte in pixel space. The conditioned RGB image is resized directly to the same `1024x1024` grid as the target; generated channels are averaged and clamped to `[0,1]` at inference.

## Conditioning ablations

- `patch`: concatenate noisy-alpha and RGB patch vectors, then use the widened patch projection.
- `pixel`: concatenate noisy-alpha and RGB channels before the pixel projection.
- `both`: use both channel-concatenation paths.
- `sequence`: project noisy-alpha and RGB patches separately with the same frozen pretrained projection, concatenate `4096 + 4096` tokens, and retain the first `4096` tokens for pixel decoding.

Only widened channel-concat projections use `[W_old/sqrt(2), W_old/sqrt(2)]`. Sequence conditioning does not widen or duplicate the pretrained projection. It uses aligned 2D RoPE by default and a zero-initialized reference-type embedding; set `--model.sequence_rope_mode=offset` to test the one-grid-offset alternative.

## Training

Activate the `pixel_dit` environment and launch from `t2i/`. The launcher uses
one visible GPU by default:

```bash
cd /home/mridul/matting/PixelDiT/t2i
bash run_matting_overfit_2gpu.sh patch
bash run_matting_overfit_2gpu.sh pixel
bash run_matting_overfit_2gpu.sh both
bash run_matting_overfit_2gpu.sh sequence
```

For one job spanning two GPUs visible in the same shell, set
`NPROC_PER_NODE=2`. Independent jobs launched from separate interactive
sessions use standalone rendezvous with automatically selected local ports, so
they can run concurrently on the same node. `MASTER_PORT` may still be set
explicitly for a managed launch.

Submit a one-H200 Slurm job with:

```bash
sbatch job_matting_overfit_1gpu.sh both
sbatch job_matting_overfit_1gpu.sh sequence
```

The scheduled launcher defaults to batch size 4 per GPU and accumulation 2
(effective batch 8). Override these without editing the script, for example:

```bash
BATCH_SIZE_PER_GPU=1 GRAD_ACCUM_STEPS=8 MAX_TRAIN_STEPS=100 \
  sbatch job_matting_overfit_1gpu.sh sequence
```

Each run uses the same deterministic category-stratified 16-image subset, batch size 1 per GPU, rank-16/alpha-16 LoRA, flow matching only, and a maximum of 1,000 optimizer steps. Compact adapters and validation outputs are written into the timestamped run directory every 100 steps. Use gradient accumulation 8 for an effective batch of 8 on one GPU, or 4 when one job spans two GPUs.

By default, every launch creates a timestamped directory under:

```text
/scratch/mridul/runs/matting/v1/pixeldit-matting-am2k-<mode>_<YYYYmmdd_HHMMSS>/
```

That directory contains the training log, deterministic subset manifest, adapter checkpoints, generated preview grids, and the local W&B run files. W&B logs the gradient-accumulated flow loss every 10 optimizer steps. Every 100 steps it logs correct/shuffled validation losses and a fixed-seed grid whose columns are `input RGB | generated alpha | ground-truth alpha`.

Every run also writes `config_resolved.yaml` after all CLI overrides are
applied. The same file is uploaded to the W&B Files tab, while batch size,
world size, accumulation, effective batch size, and used/available dataset
counts are stored under the W&B `runtime` config.

The preview uses one example and a 20-step sampler by default. These can be changed without touching the training setup:

```bash
bash run_matting_overfit_2gpu.sh both \
  --train.wandb_num_examples=2 \
  --train.wandb_sampling_steps=30
```

Override the run root or exact run directory with `MATTING_RUN_ROOT` or `MATTING_RUN_DIR`. Set `WANDB_MODE=offline` before launching when the compute node cannot reach W&B.

Resume a run with:

```bash
MATTING_RUN_DIR=/scratch/mridul/runs/matting/v1/my_resumed_both_run \
  bash run_matting_overfit_2gpu.sh both \
  --resume_from=/path/to/adapters/step_500.pth
```

## Conditioned inference and comparison

Each training run writes its exact 16-image selection to `overfit_manifest.json`. Use that manifest directly:

```bash
python matting_inference.py \
  --config configs/PixelDiT_1024px_matting_am2k_overfit.yaml \
  --adapter_path /path/to/timestamped_run/adapters/latest.pth \
  --input /path/to/timestamped_run/overfit_manifest.json \
  --output_dir /path/to/timestamped_run/correct

python matting_inference.py \
  --config configs/PixelDiT_1024px_matting_am2k_overfit.yaml \
  --adapter_path /path/to/timestamped_run/adapters/latest.pth \
  --input /path/to/timestamped_run/overfit_manifest.json \
  --output_dir /path/to/timestamped_run/shuffled \
  --shuffle_conditions
```

Inference writes float32 `.npy` alpha mattes, grayscale PNG previews, and a manifest recording the exact condition/seed pairing. Evaluate the correct and shuffled outputs with:

```bash
python evaluate_matting.py \
  --pred_dir /path/to/timestamped_run/correct \
  --shuffled_pred_dir /path/to/timestamped_run/shuffled \
  --split train --overfit_samples 16
```

The evaluator reports whole-image SAD, MSE, MAD, gradient, and connectivity errors. The pilot succeeds when training loss decreases, correct-condition MSE is below shuffled-condition MSE, output alpha changes when RGB conditions are shuffled, and reloaded adapters reproduce identical outputs for a fixed seed.

## Smoke tests

```bash
cd /home/mridul/matting/PixelDiT
python -m unittest discover -s t2i/tests -v
```

# PixelDiT AM-2K matting pilot

This pilot fine-tunes PixelDiT to generate a three-channel copy of an alpha matte in pixel space. The conditioned RGB image is resized directly to the same `1024x1024` grid as the target; generated channels are averaged and clamped to `[0,1]` at inference.

## Conditioning ablations

The reference image can enter through three independent doors, and a mode is a
choice of which are open:

- `patch`: concatenate noisy-alpha and RGB patch vectors, then use the widened patch projection.
- `pixel`: concatenate noisy-alpha and RGB channels before the pixel projection.
- `both`: use both channel-concatenation paths.
- `sequence`: project noisy-alpha and RGB patches separately with the same frozen pretrained projection, concatenate `4096 + 4096` tokens, and retain the first `4096` tokens for pixel decoding.
- `sequence_pixel`: sequence conditioning on the patch stream **and** channel concatenation on the pixel stream. Plain `sequence` leaves the pixel decoder with no access to the reference at all, which matters for a task decided by sub-patch detail.

### Widened-projection initialization

`--model.conditioning_proj_init` controls how `patch`/`pixel`/`both`/`sequence_pixel`
grow their pretrained input projections:

- `zero` (default): `[W_old, 0]`. The widened layer computes exactly `W_old @ x`
  at step 0, so the model starts bit-identical to the pretrained checkpoint and
  the conditioning pathway grows from zero, like the zero-initialized LoRA `B`
  matrices.
- `balanced`: the legacy `[W_old/sqrt(2), W_old/sqrt(2)]`. This preserves
  activation *scale* but not the *function* — the layer computes
  `W_old @ (x + c)/sqrt(2)` while every downstream block was pretrained on
  `W_old @ x`. Training's first few hundred steps go to undoing that, which is
  why `both` started near loss 1.05 while `sequence` (which widens nothing)
  started near 0.15. Keep it only to reproduce the older runs.

### Sequence-mode stream separation

Sequence conditioning does not widen or duplicate the pretrained patch
projection, and it stays frozen: both streams are natural-image-like inputs the
pretrained projection already handles, so there is nothing new to fit.

With `sequence_rope_mode=aligned` (the default) the target and reference blocks
receive *identical* RoPE positions and pass through that one shared projection,
so the learned target/reference type embeddings carry the only signal that
separates them. Both are therefore initialized at `std=0.02` rather than zero —
at zero the split is degenerate at step 0 and the model can only route on
content, which works at high noise levels and fails at the low-noise steps that
decide sample quality. Set `--model.sequence_rope_mode=offset` to test the
Kontext-style one-grid-offset alternative instead.

## Flow regime

`--scheduler.deterministic_flow` (default `true`) decides what the model sees
as input.

- `true`: the model input is pinned to exactly zero at the top of the schedule
  and the flow target reduces to `-x_start`, so training is a single-step
  regression and inference is one forward pass with `x_start = -v`. No sampler,
  no seed; `train.wandb_sampling_steps` and `scheduler.flow_shift` do not apply.
- `false`: the original stochastic regime -- random timestep, real noise, and a
  20-step DPM-solver at preview time.

The stochastic regime turned out to be the reason the earlier runs looked like
they were training while producing nothing usable. An alpha matte is a
near-binary, smooth, extremely low-complexity target, so `x_t` at the sampled
timesteps still identifies which matte it came from. "Recognize the noisy matte,
recall the matte" drives the flow loss to ~0.002 without ever reading the
conditioning image, and sampling then starts from pure noise where that
recognition signal does not exist. Measured on the 4,000-step `both` run:
between step 1,500 and 3,900 the training loss improved 2.3x while the generated
MSE got 2.2x worse and the correct-vs-shuffled conditioning gap decayed from 44%
to 13%. Every checkpoint except that run's step 1,500 was at or worse than
predicting a constant image (all-black MSE 0.266, constant-mean MSE 0.193).

Pinning the input to zero removes the shortcut by construction rather than
discouraging it: with no signal in the input, the conditioning image is the only
thing that carries information about the answer. Note that the input must be
*exactly* zero, not merely small -- at the top timestep `q_sample` still leaks
`alpha = 2.5e-4` of the target, and a scaled copy of the target is still the
target once a normalization layer has rescaled it.

## Training

Activate the `pixel_dit` environment and launch from `t2i/`. The launcher uses
one visible GPU by default:

```bash
cd /home/mridul/matting/PixelDiT/t2i
bash run_matting_overfit_2gpu.sh patch
bash run_matting_overfit_2gpu.sh pixel
bash run_matting_overfit_2gpu.sh both
bash run_matting_overfit_2gpu.sh sequence
bash run_matting_overfit_2gpu.sh sequence_pixel
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

The evaluator reports whole-image SAD, MSE, MAD, gradient, and connectivity errors. The pilot succeeds when correct-condition MSE is below shuffled-condition MSE, output alpha changes when RGB conditions are shuffled, and reloaded adapters reproduce identical outputs. Judge runs on `validation/generated_mse` and the correct-vs-shuffled gap, never on training loss alone: those two diverged for thousands of steps in the stochastic runs. Treat a conditioning gap under ~10% as a failed run rather than an early one, and compare `generated_mse` against the trivial baselines (all-black 0.266, constant-mean 0.193) before calling a result good.

## Smoke tests

```bash
cd /home/mridul/matting/PixelDiT
python -m unittest discover -s t2i/tests -v
```

# PixelDiT matting pilot (AM-2K and Distinctions-646)

This pilot fine-tunes PixelDiT to generate a three-channel copy of an alpha matte in pixel space. The conditioned RGB image is resized directly to the same `1024x1024` grid as the target; generated channels are averaged and clamped to `[0,1]` at inference.

## Datasets

Two are wired up. Pick one with `MATTING_DATASET`:

```bash
MATTING_DATASET=am2k bash run_matting_overfit_2gpu.sh both    # default
MATTING_DATASET=d646 bash run_matting_overfit_2gpu.sh both
```

Each selects its config and runs its own setup script before launching.

| | AM-2K | Distinctions-646 |
| --- | --- | --- |
| content | 2,000 natural animal photographs | 646 foregrounds composited over many backgrounds |
| soft pixels | 0.7–3.9% of frame | much higher — this is the point |
| transparency | none; water is labelled background | glass, water, veils, smoke, fine hair |
| layout | `train/original` + `train/mask` | `Train_comp/merged` + `Train_comp/alpha` |
| splits | `train` (1800), `validation` (200) | `train`, `test` |

**Why D-646 matters.** AM-2K mattes are near-binary animal silhouettes, so a
model can score `generated_mse` 0.0002 there while predicting nothing useful for
partial coverage — the soft-alpha buckets in `matting_report.py` showed exactly
that, with MAD 30–56x higher in the soft region than in the hard one. D-646 is
where that failure becomes visible in the headline number rather than only in a
diagnostic.

D-646 ships pre-composited, in the same layout Edit2Perceive's
`data_split/Distinctions_matting` lists use, so `Distinctions646MattingDataset`
is a plain paired reader like the AM-2K one; no foreground/background
compositing happens at load time.

One difference that matters: training composites are named
`<foreground>.png_<k>`, so tens of thousands of composites come from only a few
hundred foregrounds. The overfit subset is stratified by **foreground identity**
rather than sampled uniformly, so 32 samples come from 32 distinct objects
instead of repeat views of a handful. (At 32 the gain is modest — a uniform draw
happened to give 30 distinct — but it grows with subset size.)

### AM-2K

AM-2K must be extracted as:

```text
<root>/am2k_split_category.json
<root>/train/original/<sample_id>.jpg        1800 pairs
<root>/train/mask/<sample_id>.png
<root>/validation/original/<sample_id>.jpg    200 pairs
<root>/validation/mask/<sample_id>.png
```

`setup_am2k_data.sh` builds and verifies exactly that, and is idempotent — about
ten seconds when the data is already in place, so it is safe to run before every
session. `run_matting_overfit_2gpu.sh` calls it automatically
(`MATTING_SKIP_DATA_SETUP=1` opts out).

```bash
bash setup_am2k_data.sh              # extract if needed, verify, refresh mtimes
bash setup_am2k_data.sh --check      # verify only, change nothing
bash setup_am2k_data.sh --force      # re-extract regardless
```

It verifies against `am2k_split_category.json` rather than counting files, so a
partial extract is caught and the missing sample ids are named. Only `original/`
and `mask/` are extracted — skipping `bg/`, `fg/` and `trimap/` cuts the
footprint from roughly 4 GB to 1.4 GB.

**Why the extracted data kept disappearing.** The AM-2K archives were built in
2021 and `unzip` preserves archive timestamps, so a plain extract stamps every
image with `mtime` 2021-07-01 — five years stale on arrival. An age-based
scratch cleanup reaps it on the next sweep while the `.zip` files, which carry
current mtimes, survive untouched. That is why the images vanished twice and the
archives never did. The script therefore `touch`es every extracted file and
directory afterwards; that step is the fix, not a nicety. Set `AM2K_ROOT` to a
`/projects` path for a copy that is not subject to scratch cleanup at all.

### Distinctions-646

```bash
bash setup_d646_data.sh              # extract if needed, verify, refresh mtimes
bash setup_d646_data.sh --check      # verify only, change nothing
bash setup_d646_data.sh --force      # re-extract regardless
```

Same contract as the AM-2K script: idempotent, verifies that every `merged/`
file has a matching `alpha/`, and re-stamps timestamps for the same
scratch-cleanup reason. `D646_ROOT` and `D646_ARCHIVE` override the paths.

**It needs a RAR extractor that is not currently installed.** Distinctions-646
ships as a RAR5 archive, and this cluster has no `unrar`, `unar`, `7z` or
`bsdtar` on `PATH`, none in the conda env, and no module that provides one. The
script looks for all four and stops with instructions if none is found. Install
one first:

```bash
conda install -c conda-forge libarchive   # provides bsdtar, reads RAR5
conda install -c conda-forge p7zip        # provides 7z
```

`pip install rarfile` is not sufficient on its own — it shells out to an
`unrar` binary.

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

## Trimap-band loss

Whole-image MSE stops being informative once the silhouette is right. Measured
on the `both` deterministic run at step 1,550: the trimap unknown band covers
8.8% of pixels and carries 96.4% of the total squared error — a 277x higher MSE
inside the band than outside — so uniform MSE hands the part that is still wrong
under a tenth of the gradient.

`train.matting_band_loss_weight` (default `1.0`) adds SAD, MSE and gradient
error restricted to that band, following Edit2Perceive's
`get_cycle_consistency_matting_loss`. Two deliberate differences: every term is
normalized by band size rather than a fixed `/1000`, so the three stay
comparable and independent of resolution and band width; and morphology runs
on-device through separable max-pooling instead of `cv2.dilate`/`cv2.erode`.

The band term is **added to** whole-image MSE, not substituted for it:

```text
L = whole-image MSE + band_scale * (band_sad + band_mse + band_grad)
```

`band_scale` ramps linearly from 0 to 1 over `matting_band_warmup_steps` (200)
and stays at 1. E2P replaces its flow loss (`return cycle_consistency_loss`
discards it); here the MSE term stays to keep the interior from drifting, which
a band mask cannot see. The band radius is resampled every step between
`matting_band_radius_min` and `_max` (10–40) so the model cannot fit one
particular dilation. It requires `deterministic_flow`, and validation
deliberately reports base MSE only, so runs stay comparable.

This was worth roughly 15x: `both_detflow` reached `generated_mse` 0.0048 at
step 2,300; `both_band_3k` reached 0.00032 at step 2,550.

## Pixel-resolution refinement head

`--model.use_refine_head=true` (default `false`) appends a small convolutional
head after the fold. It is the **only** module in the network that couples
neighbouring pixels across a patch seam.

Everything else is either per-pixel or per-patch: `pixel_embedder` is a per-pixel
`Linear` plus a positional embedding, `compress_to_attn` collapses each 16x16
patch to one 1152-d token, attention runs at patch resolution,
`expand_from_attn` writes it back, the PiT MLP sees one pixel at a time, and
`final_layer` is a 67-parameter `RMSNorm(16) + Linear(16 -> 3)`. Two pixels that
touch across a patch boundary can only reach each other by routing up through
both patch tokens. Folding prediction error onto `(y % 16, x % 16)` shows 13.7%
variation across in-patch positions against 2.7% at a control period of 15, and
a 2.11x max/min ratio: the residual is locked to the patch lattice.

The head folds the 16-d pixel features (not just the 3-channel projection of
them), concatenates the coarse output and the full-resolution conditioning image
— 22 channels in — and adds its result residually. Feeding it the RGB is what
makes it a guided filter rather than a blur: the whisker is already present in
the conditioning image at pixel resolution, and this is the only place where
full-resolution RGB and full-resolution alpha meet with true 2-D adjacency.

At the default `refine_head_width=64` and `refine_head_dilations=[1,2,4,1]` it is
125,251 parameters (0.01% of 1.3B) with a **measured 19x19 receptive field** —
wider than a 16px patch, so every pixel reaches across at least one seam in each
direction. Use `[1,2,4,8,1]` for 35x35.

The output convolution is zero-initialized, so at step 0 the model reproduces
its pretrained output bit-for-bit and only the output convolution receives
gradient — the same discipline as `conditioning_proj_init=zero` and LoRA's zero
`B`. After one optimizer step the whole stack trains. The head has no pretrained
weights to adapt, so it trains in full rather than through LoRA; its settings are
written into adapter metadata and `matting_inference.py` rebuilds the same
architecture before loading.

Note this is a *granularity* change, not a capacity change. Fully fine-tuning
the pixel blocks (105.1M, versus about 1.4M trainable under LoRA r16) and
stacking more `PiTBlock`s (47.8M each) are both worth trying, but every
`PiTBlock` has the same 16x16 ceiling, so neither addresses the periodicity.

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
/scratch/mridul/runs/matting/v2/pixeldit-matting-am2k-<mode>_<YYYYmmdd_HHMMSS>/
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
MATTING_RUN_DIR=/scratch/mridul/runs/matting/v2/my_resumed_both_run \
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

## Full-subset report

The W&B preview shows four samples, which is too few to judge a run.
`matting_report.py` runs the adapter over every sample it trained on and writes
contact sheets plus the numbers the preview cannot show:

```bash
python matting_report.py \
  --config configs/PixelDiT_1024px_matting_am2k_overfit.yaml \
  --adapter_path /scratch/mridul/runs/matting/v2/<run>/adapters/latest.pth \
  --output_dir  /scratch/mridul/runs/matting/v2/<run>/report

# Distinctions-646
python matting_report.py --dataset d646 \
  --config configs/PixelDiT_1024px_matting_d646_overfit.yaml \
  --adapter_path /scratch/mridul/runs/matting/v2/<run>/adapters/latest.pth \
  --output_dir  /scratch/mridul/runs/matting/v2/<run>/report
```

It defaults to the exact training subset, which each run records in
`metadata["subset_sample_ids"]`; pass `--split validation` or `--sample_ids` for
held-out samples.

**Run it in fp32, which is now the default.** `Accelerator(mixed_precision=bf16)`
keeps master weights in fp32 and only casts per-op inside autocast, and the W&B
preview then calls `_sample_training_grid(accelerator.unwrap_model(model), ...)`
with no autocast context — so the preview numbers are pure fp32 with fp32
weights. Casting the weights to bf16, which these scripts used to do, is a
different computation and **collapses this model to a constant matte**: measured
MSE 0.41 and 0.51 across the 32-sample subset on two checkpoints whose fp32 MSE
is around 0.0002, with predicted alpha flat at 0.83 and 0.20 regardless of the
target. `--dtype bfloat16` reproduces that failure for A/B and warns; `--autocast`
runs the forward under autocast with fp32 weights, which is faster and close but
not identical to the preview.

Output — two images per sample, so a 32-sample run writes 64:

- `compare/<sample_id>.png` — `input RGB | generated alpha | ground truth`, at
  native resolution. Deliberately the same three columns, order, padding and
  white pad colour as the W&B preview grid, so a sample can be held against the
  training preview without accounting for layout differences.
- `alpha/<sample_id>.png` — the predicted matte on its own, full resolution
  (`--save_full_res` adds float32 `.npy`).
- `contact_sheet_NN.jpg` — eight samples per sheet, each row
  `RGB | ground truth | prediction | error heatmap | cutout on a checkerboard`.
  The checkerboard column is the one to look at for partial coverage: a matte
  that is silently binary looks fine in the alpha column and wrong here.
- `report.json` — per-sample and aggregate metrics.

`--compare_tile` shrinks the comparison images if 3x1024 per file is more than
you want to scroll.

The subset is the training subset, exactly and in order — verified against
`AM2KMattingDataset`'s own stratified selection — and the W&B preview draws its
four examples from the head of that same list. A sample's numbers here are
directly comparable to what the preview showed.

Two aggregates are worth more than the means:

**Error by ground-truth alpha.** Soft pixels are the entire matting problem and
only a few percent of the frame, so whole-image means hide them. The report
buckets error by ground-truth alpha (`background`, `near-transparent`, `half`,
`near-opaque`, `foreground`) and weights per-image means by pixel count.

**Patch-grid periodicity.** Blockiness is a claim about periodicity, so the
report measures it: mean `|error|` folded onto `(y % patch, x % patch)` against
a control period of `patch - 1`. Validated against a planted lattice — clean
noise gives 1.09% at period versus 1.11% at control, a planted 16px lattice
gives 120% versus 3.3%. A run whose period figure has fallen to its control
figure has no patch lattice left.

Note on transparency: **AM-2K contains no glass or water.** It is Animal Matting
2K — 20 animal categories, 90 images each. The only partial coverage in it is fur
and whisker edges, which is what the soft-alpha buckets measure. Run
`--dataset d646` for genuinely transparent subjects.

## Run log

`generated_mse` at the last logged step, from each run's `wandb-summary.json`.
Trivial baselines for scale: all-black 0.266, constant-mean 0.193.

| run | mode | flow | band loss | step | `generated_mse` |
| --- | --- | --- | --- | ---: | ---: |
| `btoh_pixel_band_3k` | both | deterministic | yes | 8210 | **0.000207** |
| `both_band_3k` | both | deterministic | yes | 2560 | 0.000316 |
| `sequence_pixel_band_3k` | sequence_pixel | deterministic | yes | 1450 | 0.00196 |
| `both_detflow_pixelloss` | both | deterministic | yes | 620 | 0.00297 |
| `both_detflow` | both | deterministic | no | 2310 | 0.00482 |
| `sequence_pixel_detflow` | sequence_pixel | deterministic | no | 1670 | 0.00762 |
| `both_100step` | both | stochastic | no | 100 | 0.126 |
| `both_20kstep_bs16_acc1` | both | stochastic | no | 4000 | 0.181 |
| `sequence_pixel_aligned_2000step` | sequence_pixel | stochastic | no | 1570 | 0.230 |
| `sequence_offset_2000step` | sequence | stochastic | no | 1850 | 0.268 |
| `both_zeroinit_20kstep` | both | stochastic | no | 420 | 0.338 |

Reading it: every stochastic run sits at or worse than the trivial baselines,
which is the conditioning-collapse result in the flow-regime section above.
Deterministic flow buys roughly two orders of magnitude; the band loss buys
another 15x on top; and length still helps after that — the same recipe at 8,210
steps is 1.5x better than at 2,560. The `both` versus `sequence_pixel` gap is
consistent but small next to either of those.

Two caveats when comparing: `btoh_pixel_band_3k` is a typo for `both`, not a
separate mode — its resolved config is `conditioning_mode: both`. And these are
32-image overfit runs, so `generated_mse` measures fit, not generalization.

## Smoke tests

```bash
cd /home/mridul/matting/PixelDiT
python -m unittest discover -s t2i/tests -v
```

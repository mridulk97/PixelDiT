# Text-to-Image Generation

PixelDiT-T2I text-to-image generation trained directly in pixel space at up to 1024×1024 resolution. Uses Gemma-2 as the text encoder and MM-DiT blocks for text-image fusion.

## Inference

```bash
cd t2i/
python inference.py \
  --config configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml \
  --model_path pixeldit_t2i_v1.pth \
  --txt_file prompts.txt \
  --custom_height 1024 --custom_width 1024 \
  --cfg_scale 2.75 --seed 2025 \
  --negative_prompt "low quality, worst quality, over-saturated, blurry, deformed, watermark" \
  --work_dir "."
```

Results are saved under `<work_dir>/vis/`. Checkpoint is **auto-downloaded** from [HuggingFace](https://huggingface.co/nvidia/PixelDiT-1300M-1024px) if not found locally.

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config` | — | Config YAML path |
| `--model_path` | — | Checkpoint `.pth` path|
| `--txt_file` | — | Text file with one prompt per line |
| `--custom_height` | (from config) | Output image height |
| `--custom_width` | (from config) | Output image width |
| `--cfg_scale` | 3.5 | Classifier-free guidance scale |
| `--step` | 50 | Number of sampling steps |
| `--seed` | 0 | Random seed |
| `--negative_prompt` | `""` | Negative prompt for CFG |
| `--work_dir` | (auto) | Output directory |


## Pre-trained Models

| Model | Params | Checkpoint |
|:---:|:---:|:---:|
| PixelDiT-T2I | 1.3B | [🤗 HuggingFace](https://huggingface.co/nvidia/PixelDiT-1300M-1024px/resolve/main/pixeldit_t2i_v1.pth) |

## Training

Training is launched via `train.sh` with a config YAML. The typical pipeline:

1. **Stage 1** — Pre-train at 512×512 from scratch (fixed resolution, with REPA loss)
2. **Stage 2** — Continue at 512×512 with multi-aspect ratio (no REPA loss)
3. **Stage 3** — Fine-tune at 1024×1024 with multi-aspect ratio (no REPA loss)

Note that at Stage 1, we do observe significantly faster convergence when using REPA loss. At Stage 2 and 3, our intuition is that the model has already been trained to a good point, so we stop using REPA loss and only use the flow matching loss. We have not done a thorough ablation on the REPA loss at higher resolution and later-stage training yet, due to the computation cost limitation. We welcome more discussions. 

### Stage 1: 512×512 Pre-training

```bash
cd t2i/
bash train.sh configs/PixelDiT_512px_pixel_diffusion_stage1.yaml \
  --data.data_dir="[/path/to/dataset1, /path/to/dataset2]" \
  --work_dir=/path/to/output \
  --name=pixeldit-t2i-512-stage1 \
  --tracker_project_name="pixeldit_t2i" \
  --train.save_model_steps=10000
```

### Stage 2: 512×512 Multi-Aspect-Ratio Fine-tuning

Load the Stage 1 checkpoint via `--load_from`. This stage enables multi-aspect ratio training while staying at 512px resolution, and removes the REPA loss:

```bash
cd t2i/
bash train.sh configs/PixelDiT_512px_pixel_diffusion_stage2.yaml \
  --data.data_dir="[/path/to/dataset1, /path/to/dataset2]" \
  --work_dir=/path/to/output \
  --name=pixeldit-t2i-512-stage2 \
  --tracker_project_name="pixeldit_t2i" \
  --train.save_model_steps=10000 \
  --load_from=/path/to/stage1_checkpoint.pth
```

### Stage 3: 1024×1024 Multi-Aspect-Ratio Fine-tuning

Load the Stage 2 checkpoint via `--load_from`. This stage scales up to 1024px resolution with multi-aspect ratio:

```bash
cd t2i/
bash train.sh configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml \
  --data.data_dir="[/path/to/dataset1, /path/to/dataset2]" \
  --work_dir=/path/to/output \
  --name=pixeldit-t2i-1024-stage3 \
  --tracker_project_name="pixeldit_t2i" \
  --train.save_model_steps=10000 \
  --load_from=/path/to/stage2_checkpoint.pth
```

### Resume Training

Use `--resume_from` to resume from a checkpoint (restores optimizer state, step count, etc.):

```bash
bash train.sh configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml \
  --resume_from=/path/to/checkpoint.pth \
  --work_dir=/path/to/output \
  --name=pixeldit-t2i-1024-stage3
```

> **`--load_from` vs `--resume_from`**: Use `--load_from` to load weights only (fresh optimizer). Use `--resume_from` to fully resume training (restores optimizer, scheduler, step count).

### Key Training Configs

| Config | Resolution | Multi-Aspect-Ratio | REPA Weight | Flow Shift | Batch Size |
|--------|:---:|:---:|:---:|:---:|:---:|
| `PixelDiT_512px_pixel_diffusion_stage1.yaml` | 512 | No | 0.5 | 3.0 | 8 |
| `PixelDiT_512px_pixel_diffusion_stage2.yaml` | 512 | Yes | 0.0 | 3.0 | 8 |
| `PixelDiT_1024px_pixel_diffusion_stage3.yaml` | 1024 | Yes | 0.0 | 4.0 | 3 |

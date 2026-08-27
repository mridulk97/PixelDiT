"""Focused AM-2K overfit trainer for image-conditioned PixelDiT matting."""

import datetime
import gc
import json
import logging
import os
import os.path as osp
import random
from dataclasses import asdict
from pathlib import Path

import pyrallis
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from torch.utils.data import DataLoader, default_collate
from torchvision.utils import make_grid, save_image

from diffusion import DPMS, Scheduler
from diffusion.data.builder import build_dataset
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder
from diffusion.model.matting_losses import matting_band_loss, sample_band_radius
from diffusion.model.lora import (
    configure_matting_trainable_parameters,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)
from diffusion.utils.config import PixDiTConfig, model_init_config
from diffusion.utils.misc import set_random_seed
from tools.download import resolve_checkpoint


LOGGER = logging.getLogger("pixeldit-matting")


def _checkpoint_state_dict(path: str):
    # This is a trusted, full PixelDiT checkpoint rather than a weights-only
    # tensor archive. Being explicit also avoids PyTorch's migration warning.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    checkpoint = checkpoint.copy()
    checkpoint.pop("pos_embed", None)
    return checkpoint


@torch.no_grad()
def _encode_fixed_prompt(config, device):
    options = config.data.extra or {}
    prompt = options.get(
        "default_prompt",
        "Transform to matting map while maintaining original composition",
    )
    tokenizer, text_encoder = get_tokenizer_and_text_encoder(
        name=config.text_encoder.text_encoder_name,
        device=device,
    )
    tokens = tokenizer(
        prompt,
        max_length=config.text_encoder.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    embeddings = text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0].detach().cpu()
    attention_mask = tokens.attention_mask.detach().cpu()
    del text_encoder, tokenizer, tokens
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prompt, embeddings, attention_mask


def _deterministic_inputs(target, train_sampling_steps):
    """Model input and timestep for the deterministic regime.

    The input is pinned to exactly zero at the top of the schedule. A noisy
    copy of the target is what lets the model reach low flow loss without ever
    reading the condition -- it can denoise or recall the matte from `x_t`
    alone. Removing that input entirely is the point: the reference image
    becomes the only thing carrying information about the answer.
    """
    timesteps = torch.full(
        (target.shape[0],),
        int(train_sampling_steps) - 1,
        device=target.device,
        dtype=torch.long,
    )
    return torch.zeros_like(target), timesteps


def _fires_on(global_step, interval, max_steps):
    """Whether a periodic action runs at this step. The final step always fires."""
    interval = max(1, int(interval))
    return global_step % interval == 0 or global_step == max_steps


def _band_loss_scale(config, global_step):
    """Weight on the band term, linearly warmed up over the configured steps."""
    weight = float(config.train.matting_band_loss_weight)
    if weight <= 0.0:
        return 0.0
    warmup = int(config.train.matting_band_warmup_steps)
    if warmup <= 0:
        return weight
    # Edit2Perceive ships a `--extra_loss_start_epoch` flag that its trainer
    # never reads, so its band loss is on from step 0. A ramp is available here
    # because the band is meaningless until the silhouette roughly exists.
    return weight * min(1.0, max(0, global_step) / float(warmup))


def _flow_loss(
    model,
    diffusion,
    target,
    model_kwargs,
    config,
    timesteps=None,
    noise=None,
    band_scale=0.0,
):
    """Flow-matching loss under either the stochastic or deterministic regime.

    Returns the loss and a dict of components for logging.
    """
    if config.scheduler.deterministic_flow:
        x_t, timesteps = _deterministic_inputs(target, config.scheduler.train_sampling_steps)
        output = model(x_t, timesteps, **model_kwargs)
        if isinstance(output, dict):
            output = output["x"]
        # The flow target is `noise - x_start`, and noise is exactly zero here,
        # so the model regresses `-target` in a single step.
        base = ((output.float() + target.float()) ** 2).mean()
        parts = {"base": base.detach()}
        if band_scale > 0.0:
            # `-output` is the predicted matte, by the same identity.
            band, terms = matting_band_loss(
                -output,
                target,
                radius=sample_band_radius(
                    config.train.matting_band_radius_min,
                    config.train.matting_band_radius_max,
                ),
                sad_weight=config.train.matting_band_sad_weight,
                mse_weight=config.train.matting_band_mse_weight,
                grad_weight=config.train.matting_band_grad_weight,
            )
            parts.update({f"band_{name}": value.detach() for name, value in terms.items()})
            parts["band_total"] = band.detach()
            return base + band_scale * band, parts
        return base, parts
    if band_scale > 0.0:
        raise ValueError(
            "train.matting_band_loss_weight requires scheduler.deterministic_flow=true; "
            "the stochastic regime predicts a velocity, not the matte itself"
        )
    if timesteps is None:
        timesteps = torch.randint(
            0,
            config.scheduler.train_sampling_steps,
            (target.shape[0],),
            device=target.device,
        ).long()
    loss = diffusion.training_losses(
        model,
        target,
        timesteps,
        noise=noise,
        model_kwargs=model_kwargs,
    )["loss"].mean()
    return loss, {"base": loss.detach()}


@torch.no_grad()
def _decode_deterministic(model, target_like, model_kwargs, train_sampling_steps):
    """Single forward pass from a zero input; `x_start = -v` when noise is zero."""
    x_t, timesteps = _deterministic_inputs(target_like, train_sampling_steps)
    output = model(x_t, timesteps, **model_kwargs)
    if isinstance(output, dict):
        output = output["x"]
    return -output


@torch.no_grad()
def _validation_losses(model, diffusion, batch, embeddings, attention_mask, config):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    num_steps = int(config.scheduler.train_sampling_steps)
    cpu_targets = batch[0]
    cpu_conditions = batch[8]
    if cpu_targets.shape[0] < 2:
        raise ValueError("Condition-shuffling validation requires at least two examples")
    generator = torch.Generator(device=device).manual_seed(2025)
    correct_losses = []
    shuffled_losses = []
    # Evaluate one sample at a time so validation never exceeds the training
    # batch's peak activation memory, especially in 8,192-token sequence mode.
    for index in range(cpu_targets.shape[0]):
        target = cpu_targets[index : index + 1].to(device)
        condition = cpu_conditions[index : index + 1].to(device)
        shuffled_index = (index + 1) % cpu_targets.shape[0]
        shuffled_condition = cpu_conditions[shuffled_index : shuffled_index + 1].to(device)
        timestep = None
        noise = None
        if not config.scheduler.deterministic_flow:
            timestep = torch.full((1,), num_steps // 2, device=device, dtype=torch.long)
            noise = torch.randn(target.shape, generator=generator, device=device, dtype=target.dtype)
        y = embeddings.to(device=device, dtype=target.dtype).unsqueeze(1)
        mask = attention_mask.to(device).unsqueeze(1).unsqueeze(1)
        # Deliberately the base loss only: the correct-vs-shuffled probe stays
        # comparable across runs with and without the band term.
        correct_losses.append(
            _flow_loss(
                model,
                diffusion,
                target,
                {"y": y, "mask": mask, "condition_image": condition},
                config,
                timesteps=timestep,
                noise=noise,
            )[0]
        )
        shuffled_losses.append(
            _flow_loss(
                model,
                diffusion,
                target,
                {"y": y, "mask": mask, "condition_image": shuffled_condition},
                config,
                timesteps=timestep,
                noise=noise,
            )[0]
        )
    correct = torch.stack(correct_losses).mean()
    shuffled = torch.stack(shuffled_losses).mean()
    model.train(was_training)
    return correct, shuffled


@torch.no_grad()
def _rotating_preview_indices(dataset_size, num_examples, global_step):
    """A different slice of the subset at each preview.

    Seeded by step, so re-running a job shows the same images at the same
    points, and drawn without replacement so one grid never repeats a sample.
    """
    count = max(1, min(int(num_examples), int(dataset_size)))
    return random.Random(int(global_step)).sample(range(int(dataset_size)), count)


def _collate_indices(dataset, indices):
    """Build a preview batch from arbitrary dataset indices."""
    return default_collate([dataset[index] for index in indices])


def _sample_training_grid(
    model,
    batch,
    embeddings,
    attention_mask,
    sample_steps,
    flow_shift,
    num_examples,
    config,
):
    """Generate fixed-seed alpha samples and arrange input/generated/GT triplets."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    target = batch[0][:num_examples].to(device)
    condition = batch[8][:num_examples].to(device)
    batch_size = target.shape[0]
    y = embeddings.to(device=device, dtype=target.dtype).unsqueeze(1).expand(
        batch_size, -1, -1, -1
    )
    mask = attention_mask.to(device).expand(batch_size, -1)
    if config.scheduler.deterministic_flow:
        # One forward pass, no sampler and no seed: the regime is deterministic
        # end to end, so `sample_steps` and `flow_shift` do not apply.
        samples = _decode_deterministic(
            model,
            target,
            {"y": y, "mask": mask, "condition_image": condition},
            config.scheduler.train_sampling_steps,
        )
    else:
        generator = torch.Generator(device=device).manual_seed(2025)
        noise = torch.randn(
            target.shape,
            generator=generator,
            device=device,
            dtype=target.dtype,
        )
        solver = DPMS(
            model.forward_with_dpmsolver,
            condition=y,
            uncondition=None,
            guidance_type="classifier-free",
            cfg_scale=1.0,
            model_type="flow",
            model_kwargs={"mask": mask, "condition_image": condition},
            schedule="FLOW",
            interval_guidance=[0, 1],
        )
        samples = solver.sample(
            noise,
            steps=int(sample_steps),
            order=2,
            skip_type="time_uniform_flow",
            method="multistep",
            flow_shift=float(flow_shift),
        )
    generated_alpha = ((samples.float() + 1.0) * 0.5).mean(dim=1).clamp(0.0, 1.0)
    target_alpha = ((target.float() + 1.0) * 0.5).mean(dim=1).clamp(0.0, 1.0)
    input_rgb = ((condition.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    tiles = []
    for index in range(batch_size):
        tiles.extend(
            [
                input_rgb[index].cpu(),
                generated_alpha[index].repeat(3, 1, 1).cpu(),
                target_alpha[index].repeat(3, 1, 1).cpu(),
            ]
        )
    grid = make_grid(tiles, nrow=3, padding=4, pad_value=1.0)
    generated_mse = torch.mean((generated_alpha - target_alpha).square()).item()
    model.train(was_training)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return grid, generated_mse


def _init_wandb(config, lora_info, prompt, resolved_config_path, run_metadata):
    if str(config.report_to).lower() not in {"wandb", "all"}:
        return None, None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("report_to=wandb requires the 'wandb' package") from error

    run = wandb.init(
        project=config.tracker_project_name,
        name=config.name,
        dir=config.work_dir,
        config=asdict(config),
        tags=["pixeldit", "matting", "am2k", config.model.conditioning_mode],
        group="pixeldit-am2k-conditioning-ablation",
    )
    run.config.update(
        {"lora": lora_info, "fixed_prompt": prompt, "runtime": run_metadata},
        allow_val_change=True,
    )
    # Keep the human-readable resolved YAML in the W&B Files tab in addition
    # to W&B's searchable structured config.
    run.save(str(resolved_config_path), base_path=config.work_dir, policy="now")
    return wandb, run


@pyrallis.wrap()
def main(config: PixDiTConfig) -> None:
    if config.model.conditioning_mode not in {"patch", "pixel", "both", "sequence", "sequence_pixel"}:
        raise ValueError(
            "Matting training requires "
            "conditioning_mode=patch|pixel|both|sequence|sequence_pixel"
        )
    if config.model.multi_scale:
        raise ValueError("The AM-2K overfit pilot uses fixed-resolution training")
    if config.train.ema_update:
        raise ValueError(
            "train_matting.py does not implement adapter-only EMA; keep train.ema_update=false "
            "for the conditioning pilot"
        )

    os.makedirs(config.work_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(osp.join(config.work_dir, "train_matting.log")),
        ],
    )
    init_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=5400))
    accelerator = Accelerator(
        mixed_precision=config.model.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        kwargs_handlers=[init_handler, DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    set_random_seed(int(config.train.seed) + accelerator.process_index)

    resolved_config_path = Path(config.work_dir) / "config_resolved.yaml"
    if accelerator.is_main_process:
        with resolved_config_path.open("w", encoding="utf-8") as stream:
            pyrallis.dump(config, stream=stream)
        LOGGER.info("Saved resolved run config to %s", resolved_config_path)
    accelerator.wait_for_everyone()

    prompt, prompt_embeddings, prompt_mask = _encode_fixed_prompt(config, accelerator.device)
    accelerator.wait_for_everyone()
    LOGGER.info("Cached fixed matting prompt: %s", prompt)

    model = build_model(
        config.model.model,
        config.train.grad_checkpointing,
        config.model.fp32_attention,
        **model_init_config(config, latent_size=config.model.image_size),
    )
    base_checkpoint = resolve_checkpoint(config.model.load_from or "pixeldit_t2i_v1.pth")
    load_result = model.load_state_dict(_checkpoint_state_dict(base_checkpoint), strict=False)
    allowed_missing = {"core.reference_type_embedding", "core.target_type_embedding"}
    # The refinement head is new, so the base checkpoint carries none of it.
    allowed_missing_prefixes = ("core.refine_head.",)
    disallowed_missing = [
        key
        for key in load_result.missing_keys
        if key not in allowed_missing and not key.startswith(allowed_missing_prefixes)
    ]
    if disallowed_missing or load_result.unexpected_keys:
        raise RuntimeError(
            f"Base checkpoint mismatch. Missing={disallowed_missing}, "
            f"unexpected={load_result.unexpected_keys}"
        )

    lora_info = configure_matting_trainable_parameters(
        model,
        rank=config.train.lora_rank,
        alpha=config.train.lora_alpha,
        dropout=config.train.lora_dropout,
    )
    LOGGER.info("LoRA configuration: %s", lora_info)
    if config.train.matting_band_loss_weight > 0.0:
        LOGGER.info(
            "Trimap-band loss: weight=%.3g (sad=%.3g mse=%.3g grad=%.3g), radius=%d-%d, warmup=%d",
            config.train.matting_band_loss_weight,
            config.train.matting_band_sad_weight,
            config.train.matting_band_mse_weight,
            config.train.matting_band_grad_weight,
            config.train.matting_band_radius_min,
            config.train.matting_band_radius_max,
            config.train.matting_band_warmup_steps,
        )
    LOGGER.info(
        "Flow regime: %s",
        "deterministic (zero input, single-step decode)"
        if config.scheduler.deterministic_flow
        else "stochastic (random timestep, multi-step sampler)",
    )

    dataset = build_dataset(
        asdict(config.data),
        resolution=config.model.image_size,
        max_length=config.text_encoder.model_max_length,
        config=config,
    )
    effective_batch_size = (
        int(config.train.train_batch_size)
        * int(config.train.gradient_accumulation_steps)
        * accelerator.num_processes
    )
    run_metadata = {
        "world_size": accelerator.num_processes,
        "batch_size_per_gpu": int(config.train.train_batch_size),
        "gradient_accumulation_steps": int(config.train.gradient_accumulation_steps),
        "effective_batch_size": effective_batch_size,
        "dataset_samples_used": len(dataset),
        "dataset_samples_available": int(getattr(dataset, "full_dataset_size", len(dataset))),
    }
    LOGGER.info(
        "Training setup: samples=%d/%d, batch/GPU=%d, GPUs=%d, accumulation=%d, effective_batch=%d",
        run_metadata["dataset_samples_used"],
        run_metadata["dataset_samples_available"],
        run_metadata["batch_size_per_gpu"],
        run_metadata["world_size"],
        run_metadata["gradient_accumulation_steps"],
        run_metadata["effective_batch_size"],
    )
    subset_records = [
        {
            "sample_id": record["sample_id"],
            "category": record["category"],
            "image": record["image_path"],
            "alpha": record["alpha_path"],
        }
        for record in dataset.dataset
    ]
    if accelerator.is_main_process:
        manifest_path = Path(config.work_dir) / "overfit_manifest.json"
        manifest_path.write_text(
            json.dumps({"images": [record["image"] for record in subset_records], "records": subset_records}, indent=2),
            encoding="utf-8",
        )
    dataloader = DataLoader(
        dataset,
        batch_size=config.train.train_batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    validation_loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False, num_workers=0)
    validation_batch = next(iter(validation_loader))

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.train.optimizer.get("lr", 1e-4)),
        betas=tuple(config.train.optimizer.get("betas", (0.9, 0.999))),
        eps=float(config.train.optimizer.get("eps", 1e-8)),
        weight_decay=float(config.train.optimizer.get("weight_decay", 0.0)),
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    resume_payload = None
    if config.resume_from:
        resume_payload = load_adapter_checkpoint(
            str(config.resume_from),
            model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            expected_conditioning_mode=config.model.conditioning_mode,
        )
        LOGGER.info("Resuming compact adapter from %s", config.resume_from)
    diffusion = Scheduler(
        str(config.scheduler.train_sampling_steps),
        noise_schedule=config.scheduler.noise_schedule,
        predict_flow_v=config.scheduler.predict_flow_v,
        learn_sigma=config.scheduler.learn_sigma and config.scheduler.pred_sigma,
        pred_sigma=config.scheduler.pred_sigma,
        snr=False,
        flow_shift=config.scheduler.flow_shift,
    )
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    wandb_api, wandb_run = (None, None)
    if accelerator.is_main_process:
        wandb_api, wandb_run = _init_wandb(
            config,
            lora_info,
            prompt,
            resolved_config_path,
            run_metadata,
        )
        if wandb_run is not None:
            LOGGER.info("Weights & Biases run: %s", wandb_run.url)

    prompt_embeddings = prompt_embeddings.pin_memory()
    prompt_mask = prompt_mask.pin_memory()
    max_steps = int(config.train.max_train_steps)
    save_every = int(config.train.adapter_save_steps)
    global_step = int((resume_payload or {}).get("step") or 0)
    epoch = int((resume_payload or {}).get("epoch") or 0)
    loss_accumulator = torch.zeros((), device=accelerator.device)
    loss_microsteps = 0
    optimizer.zero_grad(set_to_none=True)

    while global_step < max_steps:
        epoch += 1
        for batch in dataloader:
            target = batch[0].to(accelerator.device, non_blocking=True)
            condition = batch[8].to(accelerator.device, non_blocking=True)
            batch_size = target.shape[0]
            y = prompt_embeddings.to(device=target.device, dtype=target.dtype).unsqueeze(1).expand(
                batch_size, -1, -1, -1
            )
            mask = prompt_mask.to(target.device).unsqueeze(1).unsqueeze(1).expand(
                batch_size, -1, -1, -1
            )
            with accelerator.accumulate(model):
                loss, loss_parts = _flow_loss(
                    model,
                    diffusion,
                    target,
                    {"y": y, "mask": mask, "condition_image": condition},
                    config,
                    band_scale=_band_loss_scale(config, global_step),
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, config.train.gradient_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            loss_accumulator += loss.detach()
            loss_microsteps += 1

            if not accelerator.sync_gradients:
                continue
            global_step += 1
            step_loss = loss_accumulator / max(loss_microsteps, 1)
            should_log = global_step == 1 or global_step % max(1, int(config.train.log_interval)) == 0
            if should_log:
                gathered_loss = accelerator.gather(step_loss).mean().item()
                if accelerator.is_main_process:
                    LOGGER.info(
                        "mode=%s step=%d loss=%.6f",
                        config.model.conditioning_mode,
                        global_step,
                        gathered_loss,
                    )
                    if wandb_run is not None:
                        payload = {
                            "train/loss": gathered_loss,
                            "train/learning_rate": lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                        }
                        payload.update(
                            {f"train/{name}": value.item() for name, value in loss_parts.items()}
                        )
                        payload["train/band_loss_scale"] = _band_loss_scale(config, global_step)
                        wandb_run.log(payload, step=global_step)
            loss_accumulator.zero_()
            loss_microsteps = 0

            should_save = _fires_on(global_step, save_every, max_steps)
            # Previews used to live inside the checkpoint branch, which meant
            # wandb_image_interval only ever fired on multiples of
            # adapter_save_steps -- setting it finer silently did nothing. The
            # two cadences are independent now, and under deterministic flow a
            # preview is a single forward pass, so a fine interval is cheap.
            should_preview = config.train.wandb_log_images and _fires_on(
                global_step, config.train.wandb_image_interval, max_steps
            )
            if should_save or should_preview:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(model)
                    validation_log = {}
                    if should_save:
                        correct, shuffled = _validation_losses(
                            unwrapped,
                            diffusion,
                            validation_batch,
                            prompt_embeddings,
                            prompt_mask,
                            config,
                        )
                        LOGGER.info(
                            "validation step=%d correct_loss=%.6f shuffled_loss=%.6f",
                            global_step,
                            correct.item(),
                            shuffled.item(),
                        )
                        validation_log.update(
                            {
                                "validation/correct_condition_flow_loss": correct.item(),
                                "validation/shuffled_condition_flow_loss": shuffled.item(),
                                "validation/condition_loss_gap": shuffled.item() - correct.item(),
                            }
                        )
                    if should_preview and wandb_run is not None:
                        try:
                            num_examples = max(1, int(config.train.wandb_num_examples))
                            preview_batch = validation_batch
                            preview_ids = list(validation_batch[6][:num_examples])
                            if config.train.wandb_preview_rotate and len(dataset) > num_examples:
                                indices = _rotating_preview_indices(
                                    len(dataset), num_examples, global_step
                                )
                                preview_batch = _collate_indices(dataset, indices)
                                preview_ids = list(preview_batch[6])
                            grid, preview_mse = _sample_training_grid(
                                unwrapped,
                                preview_batch,
                                prompt_embeddings,
                                prompt_mask,
                                config.train.wandb_sampling_steps,
                                config.scheduler.flow_shift,
                                num_examples,
                                config,
                            )
                            if preview_batch is validation_batch:
                                generated_mse = preview_mse
                            else:
                                # Keep the headline metric on the fixed batch so
                                # its curve stays comparable across steps and
                                # runs; a rotating one would move with whichever
                                # samples happened to be drawn.
                                _, generated_mse = _sample_training_grid(
                                    unwrapped,
                                    validation_batch,
                                    prompt_embeddings,
                                    prompt_mask,
                                    config.train.wandb_sampling_steps,
                                    config.scheduler.flow_shift,
                                    num_examples,
                                    config,
                                )
                                validation_log["validation/preview_generated_mse"] = preview_mse
                            preview_dir = Path(config.work_dir) / "previews"
                            preview_dir.mkdir(parents=True, exist_ok=True)
                            preview_path = preview_dir / f"step_{global_step}.png"
                            save_image(grid, preview_path)
                            validation_log["validation/generated_mse"] = generated_mse
                            validation_log["validation/examples"] = wandb_api.Image(
                                str(preview_path),
                                caption=(
                                    "Columns: input RGB | generated alpha | ground-truth alpha"
                                    f" -- {', '.join(str(name) for name in preview_ids)}"
                                ),
                            )
                        except torch.cuda.OutOfMemoryError:
                            LOGGER.exception(
                                "Skipping W&B image sampling at step %d after a CUDA OOM",
                                global_step,
                            )
                            unwrapped.train()
                            torch.cuda.empty_cache()
                    if wandb_run is not None and validation_log:
                        wandb_run.log(validation_log, step=global_step)
                    if should_save:
                        adapter_dir = osp.join(config.work_dir, "adapters")
                        adapter_path = osp.join(adapter_dir, f"step_{global_step}.pth")
                        metadata = {
                            **lora_info,
                            "base_checkpoint": osp.abspath(base_checkpoint),
                            "image_size": config.model.image_size,
                            "patch_size": unwrapped.core.patch_size,
                            "sequence_rope_mode": config.model.sequence_rope_mode,
                            "sequence_rope_offset": config.model.sequence_rope_offset,
                            "use_sequence_type_embedding": config.model.use_sequence_type_embedding,
                            "conditioning_proj_init": config.model.conditioning_proj_init,
                            "use_refine_head": config.model.use_refine_head,
                            "refine_head_width": config.model.refine_head_width,
                            "refine_head_dilations": list(config.model.refine_head_dilations),
                            "deterministic_flow": config.scheduler.deterministic_flow,
                            "matting_band_loss_weight": config.train.matting_band_loss_weight,
                            "prompt": prompt,
                            "subset_sample_ids": [record["sample_id"] for record in subset_records],
                            "run_name": config.name,
                            "wandb_run_id": wandb_run.id if wandb_run is not None else None,
                        }
                        save_adapter_checkpoint(
                            adapter_path,
                            unwrapped,
                            metadata,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            step=global_step,
                            epoch=epoch,
                        )
                        latest = Path(adapter_dir) / "latest.pth"
                        if latest.exists() or latest.is_symlink():
                            latest.unlink()
                        latest.symlink_to(Path(adapter_path).resolve())
                accelerator.wait_for_everyone()

            if global_step >= max_steps:
                break

    accelerator.wait_for_everyone()
    LOGGER.info("Finished %s overfit run at step %d", config.model.conditioning_mode, global_step)
    if accelerator.is_main_process and wandb_run is not None:
        wandb_run.summary["final_step"] = global_step
        wandb_run.finish()
    accelerator.wait_for_everyone()
    # torchrun initializes a process group even for a one-GPU job. Let
    # Accelerate tear it down explicitly so NCCL does not warn at interpreter
    # shutdown and multi-GPU ranks cannot race one another during cleanup.
    accelerator.end_training()


if __name__ == "__main__":
    main()

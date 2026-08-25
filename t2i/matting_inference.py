"""Conditioned PixelDiT inference for RGB-to-alpha matting."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pyrallis
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from diffusion import DPMS
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder
from diffusion.model.lora import (
    configure_matting_trainable_parameters,
    load_trainable_state_dict,
)
from diffusion.model.utils import get_weight_dtype
from diffusion.utils.config import PixDiTConfig, model_init_config
from tools.download import resolve_checkpoint


DEFAULT_PROMPT = "Transform to matting map while maintaining original composition"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Matting YAML used for training")
    parser.add_argument("--adapter_path", required=True, help="Compact matting adapter checkpoint")
    parser.add_argument("--model_path", default=None, help="Base PixelDiT checkpoint override")
    parser.add_argument("--input", required=True, help="RGB image, directory, .txt, or .json manifest")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--shuffle_conditions",
        action="store_true",
        help="Roll RGB conditions by one sample while retaining output names",
    )
    return parser.parse_args()


def _read_config(path: str) -> PixDiTConfig:
    with open(path, "r", encoding="utf-8") as handle:
        return pyrallis.load(PixDiTConfig, handle)


def _base_state_dict(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = state_dict.copy()
    state_dict.pop("pos_embed", None)
    return state_dict


def _manifest_paths(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
    if path.suffix.lower() == ".txt":
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        return [Path(line) for line in lines if line and not line.startswith("#")]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("images", payload.get("inputs", list(payload.keys())))
        if not isinstance(payload, list):
            raise ValueError("JSON manifest must be a list or contain an 'images'/'inputs' list")
        paths = []
        for item in payload:
            value = item.get("image", item.get("condition_image")) if isinstance(item, dict) else item
            if value is None:
                raise ValueError(f"Manifest entry has no image path: {item!r}")
            paths.append(Path(value))
        return paths
    return [path]


def _resolve_input_paths(value: str) -> List[Path]:
    source = Path(value).expanduser()
    paths = _manifest_paths(source)
    if not paths:
        raise RuntimeError(f"No input images found under {source}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input images: {missing[:5]}")
    stems = [path.stem for path in paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Input image stems must be unique because they are used as output names")
    return paths


@torch.inference_mode()
def _encode_prompt(config, prompt: str, device: torch.device):
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
    embeddings = text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0][:, None]
    mask = tokens.attention_mask
    embeddings = embeddings.detach()
    mask = mask.detach()
    del tokenizer, text_encoder, tokens
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embeddings, mask


def _condition_tensor(path: Path, image_size: int, device, dtype):
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return (TF.to_tensor(image) * 2.0 - 1.0).unsqueeze(0).to(device=device, dtype=dtype)


def _apply_adapter_metadata(config, metadata):
    required_mode = metadata.get("conditioning_mode")
    if required_mode not in {"patch", "pixel", "both", "sequence"}:
        raise RuntimeError(f"Adapter has invalid conditioning_mode={required_mode!r}")
    config.model.conditioning_mode = required_mode
    config.model.sequence_rope_mode = metadata.get("sequence_rope_mode", "aligned")
    config.model.sequence_rope_offset = metadata.get("sequence_rope_offset")
    config.model.use_sequence_type_embedding = metadata.get("use_sequence_type_embedding", True)
    if int(metadata.get("image_size", config.model.image_size)) != int(config.model.image_size):
        raise RuntimeError("Adapter and inference config image sizes do not match")


@torch.inference_mode()
def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but CUDA is not available")

    adapter = torch.load(args.adapter_path, map_location="cpu")
    if "adapter_state_dict" not in adapter or "metadata" not in adapter:
        raise RuntimeError(f"{args.adapter_path} is not a PixelDiT matting adapter")
    metadata = adapter["metadata"]
    config = _read_config(args.config)
    _apply_adapter_metadata(config, metadata)

    device = torch.device(args.device)
    weight_dtype = get_weight_dtype(config.model.mixed_precision)
    base_path = resolve_checkpoint(args.model_path or metadata.get("base_checkpoint") or "pixeldit_t2i_v1.pth")
    model = build_model(
        config.model.model,
        use_grad_checkpoint=False,
        use_fp32_attention=config.model.fp32_attention,
        **model_init_config(config, latent_size=config.model.image_size),
    )
    result = model.load_state_dict(_base_state_dict(base_path), strict=False)
    allowed_missing = {"core.reference_type_embedding"}
    missing = [name for name in result.missing_keys if name not in allowed_missing]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"Base checkpoint mismatch: missing={missing}, unexpected={result.unexpected_keys}")

    configure_matting_trainable_parameters(
        model,
        rank=int(metadata["rank"]),
        alpha=float(metadata["alpha"]),
        dropout=float(metadata.get("dropout", 0.0)),
    )
    load_trainable_state_dict(model, adapter["adapter_state_dict"])
    del adapter
    model.eval().requires_grad_(False).to(device=device, dtype=weight_dtype)

    prompt = metadata.get("prompt", DEFAULT_PROMPT)
    caption_embeddings, caption_mask = _encode_prompt(config, prompt, device)
    caption_embeddings = caption_embeddings.to(dtype=weight_dtype)

    target_paths = _resolve_input_paths(args.input)
    condition_paths = target_paths[-1:] + target_paths[:-1] if args.shuffle_conditions else target_paths
    if args.shuffle_conditions and len(target_paths) < 2:
        raise ValueError("--shuffle_conditions requires at least two input images")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for index, (target_path, condition_path) in enumerate(zip(target_paths, condition_paths)):
        condition = _condition_tensor(condition_path, config.model.image_size, device, weight_dtype)
        generator = torch.Generator(device=device).manual_seed(args.seed + index)
        noise = torch.randn(
            1,
            3,
            config.model.image_size,
            config.model.image_size,
            generator=generator,
            device=device,
            dtype=weight_dtype,
        )
        solver = DPMS(
            model.forward_with_dpmsolver,
            condition=caption_embeddings,
            uncondition=None,
            guidance_type="classifier-free",
            cfg_scale=1.0,
            model_type="flow",
            model_kwargs={"mask": caption_mask, "condition_image": condition},
            schedule="FLOW",
            interval_guidance=[0, 1],
        )
        sample = solver.sample(
            noise,
            steps=args.steps,
            order=2,
            skip_type="time_uniform_flow",
            method="multistep",
            flow_shift=config.scheduler.flow_shift,
        )
        alpha = ((sample.float() + 1.0) * 0.5).mean(dim=1)[0].clamp(0.0, 1.0).cpu().numpy()
        npy_path = output_dir / f"{target_path.stem}.npy"
        png_path = output_dir / f"{target_path.stem}.png"
        np.save(npy_path, alpha.astype(np.float32, copy=False))
        Image.fromarray(np.round(alpha * 255.0).astype(np.uint8), mode="L").save(png_path)
        records.append(
            {
                "sample": target_path.stem,
                "input": str(target_path.resolve()),
                "condition": str(condition_path.resolve()),
                "seed": args.seed + index,
                "npy": str(npy_path.resolve()),
                "png": str(png_path.resolve()),
            }
        )
        print(f"[{index + 1}/{len(target_paths)}] {target_path.name} -> {png_path}", flush=True)

    manifest = {
        "conditioning_mode": metadata["conditioning_mode"],
        "sequence_rope_mode": metadata.get("sequence_rope_mode"),
        "base_checkpoint": str(base_path),
        "adapter": str(Path(args.adapter_path).resolve()),
        "shuffled_conditions": bool(args.shuffle_conditions),
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Small, dependency-free LoRA utilities for PixelDiT matting."""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap an ``nn.Linear`` with a zero-initialized low-rank update."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base_layer.out_features, bias=False)
        self.lora_A.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        self.lora_B.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        update = self.lora_B(self.lora_A(self.dropout(inputs)))
        return base + update.to(base.dtype) * self.scaling


_PATCH_TARGETS = (
    re.compile(r"^core\.patch_blocks\.\d+\.attn\.(qkv_x|proj_x)$"),
    re.compile(r"^core\.patch_blocks\.\d+\.mlp_x\.(w1|w2|w3)$"),
    re.compile(r"^core\.patch_blocks\.\d+\.adaLN_modulation_img\.0$"),
)

_PIXEL_TARGETS = (
    re.compile(r"^core\.pixel_blocks\.\d+\.(compress_to_attn|expand_from_attn)$"),
    re.compile(r"^core\.pixel_blocks\.\d+\.attn\.(qkv|proj)$"),
    re.compile(r"^core\.pixel_blocks\.\d+\.mlp\.(fc1|fc2)$"),
    re.compile(r"^core\.pixel_blocks\.\d+\.adaLN_modulation\.0$"),
)


def _matches_target(name: str) -> bool:
    return any(pattern.fullmatch(name) for pattern in (*_PATCH_TARGETS, *_PIXEL_TARGETS))


def _parent_and_leaf(root: nn.Module, qualified_name: str):
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def inject_matting_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> Iterable[str]:
    """Inject LoRA into the PixelDiT image and pixel pathways."""
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _matches_target(name)
    ]
    if not targets:
        raise RuntimeError("No PixelDiT matting LoRA target modules were found")
    for name in targets:
        parent, leaf = _parent_and_leaf(model, name)
        base_layer = getattr(parent, leaf)
        setattr(parent, leaf, LoRALinear(base_layer, rank=rank, alpha=alpha, dropout=dropout))
    return targets


def configure_matting_trainable_parameters(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> Dict[str, Any]:
    """Freeze the base model, inject LoRA, and unfreeze task-specific heads."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    lora_targets = list(inject_matting_lora(model, rank=rank, alpha=alpha, dropout=dropout))
    core = model.core
    mode = core.conditioning_mode

    # Only a widened projection needs training: it has newly initialized input
    # columns that no pretrained weight covers. Sequence modes reuse the
    # pretrained patch projection unchanged for both streams, so it stays
    # frozen and the stream distinction is carried by the type embeddings.
    if core.patch_conditioning:
        for parameter in core.s_embedder.proj.parameters():
            parameter.requires_grad = True
    if core.pixel_conditioning:
        for parameter in core.pixel_embedder.proj.parameters():
            parameter.requires_grad = True
    for type_embedding in (core.target_type_embedding, core.reference_type_embedding):
        if type_embedding is not None:
            type_embedding.requires_grad = True
    for parameter in core.final_layer.parameters():
        parameter.requires_grad = True
    # The refinement head has no pretrained weights to adapt, so it trains in
    # full. inject_matting_lora only wraps nn.Linear, so its convolutions are
    # untouched, and trainable_state_dict picks them up from requires_grad.
    refine_head = getattr(core, "refine_head", None)
    if refine_head is not None:
        for parameter in refine_head.parameters():
            parameter.requires_grad = True

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    return {
        "conditioning_mode": mode,
        "refine_head": None
        if refine_head is None
        else {
            "parameters": sum(p.numel() for p in refine_head.parameters()),
            "in_channels": refine_head.in_channels,
            "width": refine_head.width,
            "dilations": refine_head.dilations,
            "receptive_field": refine_head.receptive_field,
        },
        "rank": int(rank),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "target_modules": lora_targets,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
    }


def trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }


def load_trainable_state_dict(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    unexpected = sorted(set(state_dict) - set(parameters))
    if unexpected:
        raise RuntimeError(f"Unexpected adapter parameters: {unexpected[:10]}")
    missing_trainable = sorted(
        name for name, parameter in parameters.items() if parameter.requires_grad and name not in state_dict
    )
    if missing_trainable:
        raise RuntimeError(f"Missing adapter parameters: {missing_trainable[:10]}")
    with torch.no_grad():
        for name, value in state_dict.items():
            target = parameters[name]
            if tuple(target.shape) != tuple(value.shape):
                raise RuntimeError(
                    f"Adapter shape mismatch for {name}: {tuple(value.shape)} vs {tuple(target.shape)}"
                )
            target.copy_(value.to(device=target.device, dtype=target.dtype))


def save_adapter_checkpoint(
    path: str,
    model: nn.Module,
    metadata: Mapping[str, Any],
    optimizer=None,
    lr_scheduler=None,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: Dict[str, Any] = {
        "adapter_state_dict": trainable_state_dict(model),
        "metadata": dict(metadata),
        "step": step,
        "epoch": epoch,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if lr_scheduler is not None:
        payload["scheduler"] = lr_scheduler.state_dict()
    torch.save(payload, path)
    return path


def load_adapter_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    lr_scheduler=None,
    map_location="cpu",
    expected_conditioning_mode: Optional[str] = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    if "adapter_state_dict" not in checkpoint:
        raise RuntimeError(f"{path} is not a PixelDiT matting adapter checkpoint")
    if expected_conditioning_mode is not None:
        actual_mode = checkpoint.get("metadata", {}).get("conditioning_mode")
        if actual_mode != expected_conditioning_mode:
            raise RuntimeError(
                f"Adapter conditioning mode {actual_mode!r} does not match "
                f"expected mode {expected_conditioning_mode!r}"
            )
    load_trainable_state_dict(model, checkpoint["adapter_state_dict"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if lr_scheduler is not None and "scheduler" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint

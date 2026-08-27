"""Full-subset qualitative and quantitative report for a matting adapter.

The W&B preview grid shows four samples, which is too few to judge a run. This
runs the adapter over every sample it was trained on, writes contact sheets you
can scroll, and reports the numbers that the preview cannot show:

* per-sample SAD / MSE / MAD / gradient / connectivity, plus the trimap-band
  versions, so a run can be ranked rather than eyeballed;
* a **soft-alpha breakdown** -- error bucketed by ground-truth alpha. Whether a
  matte handles partial coverage at all lives entirely in ``0 < alpha < 1``,
  and whole-image means hide it because those pixels are a few percent of the
  frame;
* the **patch-grid diagnostic** -- prediction error folded onto
  ``(y % patch, x % patch)`` against a control period. Blockiness is a claim
  about periodicity, and this measures it instead of squinting at a preview.

Usage
-----
    python matting_report.py \\
        --config configs/PixelDiT_1024px_matting_am2k_overfit.yaml \\
        --adapter_path /scratch/mridul/runs/matting/v2/<run>/adapters/latest.pth \\
        --output_dir  /scratch/mridul/runs/matting/v2/<run>/report

By default it reports on exactly the samples the adapter trained on: the run
records them in ``metadata["subset_sample_ids"]``. Pass a held-out split
(``--split validation`` for AM-2K, ``--split test`` for D-646) or explicit
``--sample_ids`` to report on other samples. ``--dataset d646`` switches to the
Distinctions-646 layout.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

from diffusion.model.builder import build_model
from diffusion.model.lora import (
    configure_matting_trainable_parameters,
    load_trainable_state_dict,
)
from diffusion.model.utils import get_weight_dtype
from diffusion.utils.config import model_init_config
from diffusion.utils.matting_metrics import compute_matting_metrics
from matting_inference import (
    DEFAULT_PROMPT,
    _apply_adapter_metadata,
    _base_state_dict,
    _encode_prompt,
    _read_config,
)
from tools.download import resolve_checkpoint


# Where each dataset keeps its RGB/alpha pairs. Both are plain paired readers:
# D-646 ships pre-composited, so nothing is composited here either.
DATASET_LAYOUTS = {
    "am2k": {
        "root": "/scratch/mridul/data/matting/am-2k",
        "splits": {"train": "train", "validation": "validation"},
        "image": ("original", ".jpg"),
        "alpha": ("mask", ".png"),
    },
    "d646": {
        "root": "/scratch/mridul/data/matting/distinctions-646",
        "splits": {"train": "Train_comp", "test": "Test_comp"},
        "image": ("merged", ".png"),
        "alpha": ("alpha", ".png"),
    },
}


# Buckets over ground-truth alpha. The middle three are the soft region: fur,
# whiskers, motion blur, and -- in a dataset that has them -- glass and water.
ALPHA_BUCKETS = (
    ("background", 0.0, 0.02),
    ("near-transparent", 0.02, 0.25),
    ("half", 0.25, 0.75),
    ("near-opaque", 0.75, 0.98),
    ("foreground", 0.98, 1.0001),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Matting YAML used for training")
    parser.add_argument("--adapter_path", required=True, help="Matting adapter checkpoint")
    parser.add_argument("--model_path", default=None, help="Base PixelDiT checkpoint override")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--dataset",
        default="am2k",
        choices=sorted(DATASET_LAYOUTS),
        help="Which dataset the adapter trained on",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Dataset root (defaults to the layout's standard path)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to pull ground truth from (am2k: train/validation, d646: train/test)",
    )
    parser.add_argument(
        "--sample_ids",
        nargs="*",
        default=None,
        help="Explicit sample ids. Defaults to the adapter's own training subset.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Cap the number of samples (0 = all)")
    parser.add_argument("--tile", type=int, default=384, help="Contact-sheet tile size in pixels")
    parser.add_argument(
        "--compare_tile",
        type=int,
        default=0,
        help="Per-sample comparison tile size (0 = native resolution)",
    )
    parser.add_argument("--rows_per_sheet", type=int, default=8)
    parser.add_argument("--band_radius", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help=(
            "Weight dtype. Keep float32: the W&B preview runs the unwrapped "
            "model in fp32, and casting the weights collapses this model to a "
            "constant matte."
        ),
    )
    parser.add_argument(
        "--autocast",
        action="store_true",
        help=(
            "Run the forward under autocast at model.mixed_precision. Faster, "
            "but the W&B preview does not do this, so numbers shift slightly."
        ),
    )
    parser.add_argument(
        "--save_full_res",
        action="store_true",
        help="Also write each sample's predicted alpha at full resolution",
    )
    return parser.parse_args()


def resolve_sample_ids(args, metadata, layout, data_root: Path) -> List[str]:
    """Prefer the ids the adapter actually trained on over re-deriving them.

    A run records its subset in the checkpoint, so a report stays correct even
    if the config or the dataset on disk changed afterwards.
    """
    if args.sample_ids:
        return list(args.sample_ids)
    recorded = metadata.get("subset_sample_ids")
    if recorded and args.split == "train":
        return list(recorded)
    split_dir = layout["splits"][args.split]
    image_dir, suffix = layout["image"]
    directory = data_root / split_dir / image_dir
    if not directory.is_dir():
        raise FileNotFoundError(f"No such split directory: {directory}")
    return sorted(path.stem for path in directory.glob(f"*{suffix}"))


def load_pair(data_root: Path, split: str, sample_id: str, size: int, layout=None):
    """Load one RGB/alpha pair exactly the way the training dataset does.

    Same resizes as the loaders -- BICUBIC for the image, BILINEAR for the
    alpha -- so a metric here is comparable to one from training.
    """
    layout = layout or DATASET_LAYOUTS["am2k"]
    split_dir = layout["splits"][split]
    image_dir, image_suffix = layout["image"]
    alpha_dir, alpha_suffix = layout["alpha"]
    image_path = data_root / split_dir / image_dir / f"{sample_id}{image_suffix}"
    alpha_path = data_root / split_dir / alpha_dir / f"{sample_id}{alpha_suffix}"
    if not image_path.is_file() or not alpha_path.is_file():
        raise FileNotFoundError(
            f"Missing pair for {sample_id}: {image_path}, {alpha_path}. "
            "Run the matching setup_*_data.sh first."
        )
    image = Image.open(image_path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    alpha = Image.open(alpha_path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return image, np.asarray(alpha, dtype=np.float32) / 255.0


def alpha_bucket_errors(prediction: np.ndarray, target: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Mean absolute error split by ground-truth alpha.

    A model can score well overall while doing nothing useful with partial
    coverage, because soft pixels are a small share of the frame. This
    separates the two.
    """
    report = {}
    for name, low, high in ALPHA_BUCKETS:
        mask = (target >= low) & (target < high)
        count = int(mask.sum())
        report[name] = {
            "pixels": count,
            "share": float(count / target.size),
            "mad": float(np.abs(prediction[mask] - target[mask]).mean()) if count else 0.0,
            "mean_predicted_alpha": float(prediction[mask].mean()) if count else 0.0,
            "mean_target_alpha": float(target[mask].mean()) if count else 0.0,
        }
    return report


def patch_grid_periodicity(error: np.ndarray, period: int, control: int) -> Dict[str, float]:
    """Fold |error| onto (y % period, x % period) and measure the structure.

    A model whose finest spatial coupling is one patch leaves error locked to
    the patch lattice. Comparing against a control period that shares no factor
    with the real one separates that from ordinary noise.
    """

    def fold(size: int) -> np.ndarray:
        height, width = error.shape
        rows = (height // size) * size
        cols = (width // size) * size
        block = error[:rows, :cols].reshape(rows // size, size, cols // size, size)
        return block.mean(axis=(0, 2))

    folded = fold(period)
    control_folded = fold(control)
    return {
        "period": period,
        "control_period": control,
        "variation_at_period": float(folded.std() / folded.mean()) if folded.mean() else 0.0,
        "variation_at_control": float(control_folded.std() / control_folded.mean())
        if control_folded.mean()
        else 0.0,
        "max_over_min": float(folded.max() / folded.min()) if folded.min() > 0 else float("inf"),
    }


def checkerboard(size: int, square: int = 32) -> np.ndarray:
    """The standard alpha backdrop: partial coverage is visible against it."""
    axis = (np.arange(size) // square) % 2
    tiles = np.logical_xor(axis[:, None], axis[None, :])
    return np.where(tiles, 0.62, 0.88).astype(np.float32)


def error_heatmap(error: np.ndarray) -> np.ndarray:
    """White-to-red ramp. Absolute scale, so sheets stay comparable."""
    intensity = np.clip(error / 0.35, 0.0, 1.0)
    return np.stack(
        [np.ones_like(intensity), 1.0 - intensity, 1.0 - intensity * 0.85], axis=-1
    )


def to_uint8(array: np.ndarray) -> np.ndarray:
    return np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8)


def build_compare(
    image: Image.Image,
    target: np.ndarray,
    prediction: np.ndarray,
    tile: int,
    padding: int = 4,
) -> Image.Image:
    """`input RGB | generated alpha | ground-truth alpha`.

    Deliberately the same three columns, order, padding and white pad colour as
    the W&B preview grid (`_sample_training_grid`), so a sample here can be held
    against the training preview without accounting for layout differences.
    """
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    tiles = [
        to_uint8(rgb),
        to_uint8(np.repeat(prediction[..., None], 3, axis=-1)),
        to_uint8(np.repeat(target[..., None], 3, axis=-1)),
    ]
    width = tile * len(tiles) + padding * (len(tiles) + 1)
    canvas = Image.new("RGB", (width, tile + 2 * padding), (255, 255, 255))
    for index, array in enumerate(tiles):
        cell = Image.fromarray(array)
        if cell.size != (tile, tile):
            cell = cell.resize((tile, tile), Image.Resampling.BILINEAR)
        canvas.paste(cell, (padding + index * (tile + padding), padding))
    return canvas


def build_panel(
    image: Image.Image,
    target: np.ndarray,
    prediction: np.ndarray,
    tile: int,
) -> Image.Image:
    """One row: RGB | ground truth | prediction | error | cutout on checkerboard."""
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    board = checkerboard(target.shape[0])
    composite = rgb * prediction[..., None] + board[..., None] * (1.0 - prediction[..., None])

    tiles = [
        to_uint8(rgb),
        to_uint8(np.repeat(target[..., None], 3, axis=-1)),
        to_uint8(np.repeat(prediction[..., None], 3, axis=-1)),
        to_uint8(error_heatmap(np.abs(prediction - target))),
        to_uint8(composite),
    ]
    panel = Image.new("RGB", (tile * len(tiles), tile), (255, 255, 255))
    for index, array in enumerate(tiles):
        panel.paste(
            Image.fromarray(array).resize((tile, tile), Image.Resampling.BILINEAR),
            (index * tile, 0),
        )
    return panel


def write_sheets(panels: Sequence[Image.Image], labels: Sequence[str], output_dir: Path, rows: int):
    """Contact sheets, `rows` panels each. Written next to the per-sample files."""
    written = []
    for start in range(0, len(panels), rows):
        chunk = panels[start : start + rows]
        width = max(panel.width for panel in chunk)
        sheet = Image.new("RGB", (width, sum(panel.height for panel in chunk)), (255, 255, 255))
        offset = 0
        for panel in chunk:
            sheet.paste(panel, (0, offset))
            offset += panel.height
        index = start // rows + 1
        path = output_dir / f"contact_sheet_{index:02d}.jpg"
        sheet.save(path, quality=92)
        written.append((path, labels[start : start + rows]))
    return written


@torch.inference_mode()
def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but CUDA is not available")

    adapter = torch.load(args.adapter_path, map_location="cpu")
    if "adapter_state_dict" not in adapter or "metadata" not in adapter:
        raise RuntimeError(f"{args.adapter_path} is not a PixelDiT matting adapter")
    metadata = adapter["metadata"]
    config = _read_config(args.config)
    _apply_adapter_metadata(config, metadata)
    if not config.scheduler.deterministic_flow:
        raise RuntimeError(
            "matting_report only supports deterministic-flow adapters; use "
            "matting_inference.py for solver-based sampling."
        )

    device = torch.device(args.device)
    image_size = int(config.model.image_size)
    # `Accelerator(mixed_precision=...)` keeps master weights in fp32 and only
    # casts per-op inside autocast. The W&B preview then runs
    # `accelerator.unwrap_model(model)` with no autocast context at all, so the
    # numbers it reports are pure fp32. Reproduce that. Casting the weights to
    # bf16 is a different computation entirely and collapses this model to a
    # constant matte -- measured at MSE 0.41-0.51 against 0.0002 in fp32.
    weight_dtype = getattr(torch, args.dtype)
    if weight_dtype is not torch.float32:
        print(
            f"warning: --dtype {args.dtype} casts the weights. The W&B preview does "
            "not, and this model collapses to a constant matte under it.",
            flush=True,
        )
    autocast_dtype = get_weight_dtype(config.model.mixed_precision) if args.autocast else None
    if autocast_dtype is torch.float32:
        autocast_dtype = None

    def forward_context():
        if autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=autocast_dtype)

    model = build_model(
        config.model.model,
        use_grad_checkpoint=False,
        use_fp32_attention=config.model.fp32_attention,
        **model_init_config(config, latent_size=image_size),
    )
    base_path = resolve_checkpoint(
        args.model_path or metadata.get("base_checkpoint") or "pixeldit_t2i_v1.pth"
    )
    result = model.load_state_dict(_base_state_dict(base_path), strict=False)
    allowed = {"core.reference_type_embedding", "core.target_type_embedding"}
    missing = [
        name
        for name in result.missing_keys
        if name not in allowed and not name.startswith("core.refine_head.")
    ]
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

    caption_embeddings, caption_mask = _encode_prompt(
        config, metadata.get("prompt", DEFAULT_PROMPT), device
    )
    caption_embeddings = caption_embeddings.to(dtype=weight_dtype)

    layout = DATASET_LAYOUTS[args.dataset]
    if args.split not in layout["splits"]:
        raise ValueError(
            f"--split {args.split!r} is not a {args.dataset} split; "
            f"expected one of {sorted(layout['splits'])}"
        )
    data_root = Path(args.data_root or layout["root"])
    sample_ids = resolve_sample_ids(args, metadata, layout, data_root)
    if args.limit > 0:
        sample_ids = sample_ids[: args.limit]
    output_dir = Path(args.output_dir)
    # Two images per sample: a W&B-shaped triptych for comparison, and the bare
    # predicted matte for looking at on its own.
    (output_dir / "compare").mkdir(parents=True, exist_ok=True)
    (output_dir / "alpha").mkdir(parents=True, exist_ok=True)
    compare_tile = args.compare_tile if args.compare_tile > 0 else image_size

    zeros = torch.zeros(1, 3, image_size, image_size, device=device, dtype=weight_dtype)
    timesteps = torch.full(
        (1,), int(config.scheduler.train_sampling_steps) - 1, device=device, dtype=torch.long
    )

    panels, labels, records = [], [], []
    accumulated_error = np.zeros((image_size, image_size), dtype=np.float64)

    for index, sample_id in enumerate(sample_ids, start=1):
        image, target = load_pair(data_root, args.split, sample_id, image_size, layout)
        condition = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
        condition = (condition.permute(2, 0, 1) * 2.0 - 1.0).unsqueeze(0)
        condition = condition.to(device=device, dtype=weight_dtype)

        # Mirror training: zero input at the top of the schedule, one forward
        # pass, and x_start = -v because the noise term is exactly zero.
        with forward_context():
            velocity = model(
                zeros, timesteps, caption_embeddings, mask=caption_mask, condition_image=condition
            )
        if isinstance(velocity, dict):
            velocity = velocity["x"]
        prediction = ((-velocity).float() + 1.0).mul_(0.5).mean(dim=1)[0].clamp_(0.0, 1.0)
        prediction = prediction.cpu().numpy()

        metrics = compute_matting_metrics(prediction, target, band_radius=args.band_radius)
        record = {
            "sample_id": sample_id,
            "metrics": metrics,
            "alpha_buckets": alpha_bucket_errors(prediction, target),
        }
        records.append(record)
        accumulated_error += np.abs(prediction - target)

        Image.fromarray(to_uint8(prediction), mode="L").save(output_dir / "alpha" / f"{sample_id}.png")
        build_compare(image, target, prediction, compare_tile).save(
            output_dir / "compare" / f"{sample_id}.png"
        )
        if args.save_full_res:
            np.save(output_dir / "alpha" / f"{sample_id}.npy", prediction.astype(np.float32))
        panels.append(build_panel(image, target, prediction, args.tile))
        labels.append(sample_id)
        print(
            f"[{index}/{len(sample_ids)}] {sample_id}  "
            f"mse={metrics['mse']:.5f}  sad={metrics['sad']:.2f}  "
            f"band_mse={metrics['band_mse']:.5f}",
            flush=True,
        )

    sheets = write_sheets(panels, labels, output_dir, args.rows_per_sheet)

    # ---- aggregates -------------------------------------------------------
    keys = sorted(records[0]["metrics"])
    summary = {key: float(np.mean([r["metrics"][key] for r in records])) for key in keys}
    buckets = {}
    for name, _, _ in ALPHA_BUCKETS:
        pixels = sum(r["alpha_buckets"][name]["pixels"] for r in records)
        if pixels == 0:
            buckets[name] = {"pixels": 0, "share": 0.0, "mad": 0.0}
            continue
        # Weight per-image means by pixel count so a picture with two soft
        # pixels does not count as much as one with two million.
        weighted = sum(
            r["alpha_buckets"][name]["mad"] * r["alpha_buckets"][name]["pixels"] for r in records
        )
        buckets[name] = {
            "pixels": pixels,
            "share": pixels / (len(records) * image_size * image_size),
            "mad": weighted / pixels,
        }

    extra = config.model.extra or {}
    patch_size = int(
        extra.get("patch_size", 16) if isinstance(extra, dict) else getattr(extra, "patch_size", 16)
    )
    periodicity = patch_grid_periodicity(accumulated_error / len(records), patch_size, patch_size - 1)

    payload = {
        "adapter": str(Path(args.adapter_path).resolve()),
        "dataset": args.dataset,
        "conditioning_mode": metadata.get("conditioning_mode"),
        "use_refine_head": metadata.get("use_refine_head", False),
        "step": metadata.get("step"),
        "split": args.split,
        "samples": len(records),
        "mean_metrics": summary,
        "alpha_buckets": buckets,
        "patch_grid": periodicity,
        "per_sample": records,
    }
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- printed summary --------------------------------------------------
    print(f"\n{len(records)} samples from the {args.dataset} {args.split} split\n")
    print("  mean metrics")
    for key in keys:
        print(f"    {key:18s} {summary[key]:.5f}")

    print("\n  error by ground-truth alpha  (soft pixels are the whole matting problem)")
    print(f"    {'bucket':18s} {'share':>8s} {'MAD':>9s}")
    for name, _, _ in ALPHA_BUCKETS:
        entry = buckets[name]
        print(f"    {name:18s} {entry['share'] * 100:7.3f}% {entry['mad']:9.5f}")
    soft = sum(buckets[n]["share"] for n, _, _ in ALPHA_BUCKETS[1:4])
    print(f"    -> soft pixels are {soft * 100:.2f}% of the frame")

    print(f"\n  patch-grid periodicity (patch_size={patch_size})")
    print(f"    variation at period {periodicity['period']:<3d} {periodicity['variation_at_period'] * 100:6.2f}%")
    print(f"    variation at control {periodicity['control_period']:<2d} {periodicity['variation_at_control'] * 100:6.2f}%")
    print(f"    max/min across positions   {periodicity['max_over_min']:.2f}x")
    print("    (a ratio near the control means the patch lattice is gone)")

    worst = sorted(records, key=lambda r: r["metrics"]["mse"], reverse=True)[:5]
    print("\n  worst five by MSE")
    for record in worst:
        print(f"    {record['sample_id']:14s} mse={record['metrics']['mse']:.5f}")

    print(f"\n  per-sample images ({2 * len(records)} files)")
    print(f"    {output_dir / 'compare'}/<id>.png   RGB | generated | ground truth")
    print(f"    {output_dir / 'alpha'}/<id>.png     predicted alpha only")
    print(f"\n  contact sheets ({args.rows_per_sheet} samples each)")
    for path, names in sheets:
        print(f"    {path}  {names[0]}...{names[-1]}")
    print(f"\n  report.json  {output_dir / 'report.json'}")


if __name__ == "__main__":
    main()

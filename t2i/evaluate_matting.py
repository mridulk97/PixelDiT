"""Evaluate PixelDiT alpha .npy files against AM-2K masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from diffusion.data.datasets.pixdit_datasets import AM2KMattingDataset
from diffusion.utils.matting_metrics import compute_matting_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--shuffled_pred_dir", default=None)
    parser.add_argument("--data_root", default="/scratch/mridul/data/matting/am-2k")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--overfit_samples", type=int, default=0)
    parser.add_argument("--overfit_seed", type=int, default=2025)
    parser.add_argument("--output", default=None, help="Optional metrics JSON path")
    return parser.parse_args()


def _load_prediction(path: Path):
    prediction = np.load(path).astype(np.float32)
    if prediction.ndim == 3:
        prediction = prediction.mean(axis=0 if prediction.shape[0] in (1, 3) else -1)
    return prediction.squeeze()


def _evaluate(prediction_dir: Path, records):
    totals = {name: 0.0 for name in ("sad", "mse", "mad", "gradient", "connectivity")}
    per_sample = []
    for record in records:
        prediction_path = prediction_dir / f"{record['sample_id']}.npy"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing prediction: {prediction_path}")
        prediction = _load_prediction(prediction_path)
        alpha = Image.open(record["alpha_path"]).convert("L")
        alpha = alpha.resize((prediction.shape[1], prediction.shape[0]), Image.Resampling.BILINEAR)
        target = np.asarray(alpha, dtype=np.float32) / 255.0
        values = compute_matting_metrics(prediction, target)
        per_sample.append({"sample_id": record["sample_id"], **values})
        for name, value in values.items():
            totals[name] += value
    mean = {name: value / len(records) for name, value in totals.items()}
    return {"count": len(records), "mean": mean, "per_sample": per_sample}


def main():
    args = parse_args()
    dataset = AM2KMattingDataset(
        data_dir=args.data_root,
        resolution=1024,
        extra={
            "split": args.split,
            "overfit_samples": args.overfit_samples,
            "overfit_seed": args.overfit_seed,
        },
    )
    result = {"correct": _evaluate(Path(args.pred_dir), dataset.dataset)}
    if args.shuffled_pred_dir:
        result["shuffled"] = _evaluate(Path(args.shuffled_pred_dir), dataset.dataset)
        result["correct_minus_shuffled_mse"] = (
            result["correct"]["mean"]["mse"] - result["shuffled"]["mean"]["mse"]
        )
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

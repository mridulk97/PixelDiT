"""Dependency-light implementations of the five common alpha-matting metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def _validate(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float32).squeeze()
    target = np.asarray(target, dtype=np.float32).squeeze()
    if prediction.ndim != 2 or target.ndim != 2 or prediction.shape != target.shape:
        raise ValueError(
            f"Expected matching [H,W] alpha arrays, got {prediction.shape} and {target.shape}"
        )
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Alpha arrays contain NaN or infinity")
    return np.clip(prediction, 0.0, 1.0), np.clip(target, 0.0, 1.0)


def _gaussian_derivative_kernels(sigma=1.4):
    epsilon = 1e-2
    half_size = int(np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))))
    coordinates = np.arange(-half_size, half_size + 1, dtype=np.float32)
    gaussian = np.exp(-(coordinates**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    derivative = -coordinates * gaussian / (sigma**2)
    kernel_x = np.outer(gaussian, derivative)
    kernel_x /= np.sqrt(np.sum(np.abs(kernel_x) ** 2))
    return kernel_x.astype(np.float32), kernel_x.T.copy().astype(np.float32)


def gradient_error(prediction, target):
    prediction, target = _validate(prediction, target)
    kernel_x, kernel_y = _gaussian_derivative_kernels()
    images = torch.from_numpy(np.stack([prediction, target]))[:, None]
    kernels = torch.from_numpy(np.stack([kernel_x, kernel_y]))[:, None]
    padding = kernel_x.shape[0] // 2
    gradients = F.conv2d(F.pad(images, (padding,) * 4, mode="replicate"), kernels)
    magnitudes = torch.sqrt(torch.sum(gradients.square(), dim=1))
    return float(torch.sum((magnitudes[0] - magnitudes[1]).square()).item() / 1000.0)


def _row_runs(mask):
    padded = np.pad(mask.astype(np.int8, copy=False), ((0, 0), (1, 1)))
    changes = np.diff(padded, axis=1)
    for row in range(mask.shape[0]):
        starts = np.flatnonzero(changes[row] == 1)
        ends = np.flatnonzero(changes[row] == -1)
        yield row, list(zip(starts.tolist(), ends.tolist()))


def _largest_connected_component(mask):
    """Return the largest 4-connected foreground component using run-length union-find."""
    mask = np.asarray(mask, dtype=bool)
    output = np.zeros_like(mask)
    parents = []
    sizes = []
    stored_runs = []

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if sizes[root_left] < sizes[root_right]:
            root_left, root_right = root_right, root_left
        parents[root_right] = root_left
        sizes[root_left] += sizes[root_right]

    previous = []
    for row, intervals in _row_runs(mask):
        current = []
        previous_index = 0
        for start, end in intervals:
            run_index = len(parents)
            parents.append(run_index)
            sizes.append(end - start)
            stored_runs.append((row, start, end, run_index))
            while previous_index < len(previous) and previous[previous_index][1] <= start:
                previous_index += 1
            overlap_index = previous_index
            while overlap_index < len(previous) and previous[overlap_index][0] < end:
                previous_start, previous_end, previous_run = previous[overlap_index]
                if previous_end > start and previous_start < end:
                    union(run_index, previous_run)
                overlap_index += 1
            current.append((start, end, run_index))
        previous = current

    if not parents:
        return output
    root_sizes = {}
    for index in range(len(parents)):
        root = find(index)
        root_sizes[root] = sizes[root]
    largest_root = max(root_sizes, key=root_sizes.get)
    for row, start, end, run_index in stored_runs:
        if find(run_index) == largest_root:
            output[row, start:end] = True
    return output


def connectivity_error(prediction, target, step=0.1):
    prediction, target = _validate(prediction, target)
    level_map = np.full_like(prediction, -1.0, dtype=np.float32)
    thresholds = np.arange(0.0, 1.0 + step, step, dtype=np.float32)
    for index in range(1, len(thresholds)):
        overlap = (prediction >= thresholds[index]) & (target >= thresholds[index])
        omega = _largest_connected_component(overlap)
        update = (level_map < 0) & ~omega
        level_map[update] = thresholds[index - 1]
    level_map[level_map < 0] = 1.0
    prediction_delta = prediction - level_map
    target_delta = target - level_map
    prediction_phi = 1.0 - prediction_delta * (prediction_delta >= 0.15)
    target_phi = 1.0 - target_delta * (target_delta >= 0.15)
    return float(np.abs(prediction_phi - target_phi).sum(dtype=np.float64) / 1000.0)


def compute_matting_metrics(prediction, target) -> Dict[str, float]:
    prediction, target = _validate(prediction, target)
    difference = prediction - target
    absolute = np.abs(difference)
    return {
        "sad": float(absolute.sum(dtype=np.float64) / 1000.0),
        "mse": float(np.mean(np.square(difference), dtype=np.float64)),
        "mad": float(np.mean(absolute, dtype=np.float64)),
        "gradient": gradient_error(prediction, target),
        "connectivity": connectivity_error(prediction, target),
    }

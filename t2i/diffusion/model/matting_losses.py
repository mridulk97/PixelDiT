"""Trimap-band matting losses for pixel-space alpha prediction.

Whole-image MSE stops being informative once the silhouette is right. Measured
on the `both` deterministic run at step 1,550: the trimap unknown band covers
8.8% of pixels and carries 96.4% of the total squared error, so uniform MSE
hands the part that is still wrong less than a tenth of the gradient. These are
the classic matting losses -- SAD, MSE, and gradient -- restricted to that band,
following Edit2Perceive's `get_cycle_consistency_matting_loss`.

Two deliberate differences from that implementation:

* Every term is normalized by the band size rather than a fixed ``/1000``.
  Theirs scales with resolution and band width, so the SAD term silently
  dominates the other two by a couple of orders of magnitude and the balance
  shifts if you change image size. Band means keep the three comparable and
  make the weights mean what they say.
* Morphology runs on-device through separable max-pooling instead of
  ``cv2.dilate``/``cv2.erode``, so the band is built in the training loop
  without a host round-trip (and without an OpenCV dependency).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


_GRADIENT_KERNEL_CACHE: Dict[Tuple[float, torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}


def alpha_from_model_space(images: torch.Tensor) -> torch.Tensor:
    """Map a 3-channel `[-1, 1]` matte to the single-channel `[0, 1]` alpha.

    The pilot predicts three copies of the alpha and averages them, which is
    also what the preview grid and the evaluator do.
    """
    if images.dim() != 4:
        raise ValueError(f"Expected [B, C, H, W], got {tuple(images.shape)}")
    return ((images + 1.0) * 0.5).mean(dim=1, keepdim=True).clamp(0.0, 1.0)


def _box_dilate(alpha: torch.Tensor, radius: int) -> torch.Tensor:
    """Grayscale dilation by a `(2r+1)` square, applied separably."""
    if radius <= 0:
        return alpha
    size = 2 * radius + 1
    dilated = F.max_pool2d(alpha, (1, size), stride=1, padding=(0, radius))
    return F.max_pool2d(dilated, (size, 1), stride=1, padding=(radius, 0))


def unknown_band(
    alpha: torch.Tensor,
    radius: int,
    foreground_threshold: float = 0.996,
    background_threshold: float = 0.004,
) -> torch.Tensor:
    """The trimap unknown region: everything a dilate/erode pair disagrees on.

    `alpha` is `[B, 1, H, W]` in `[0, 1]`. Returns a float mask of the same
    shape. This is the region a trimap would mark 128; the thresholds mirror
    Edit2Perceive's `gen_trimap`, which treats 254/255 as certain foreground and
    1/255 as certain background.
    """
    dilated = _box_dilate(alpha, radius)
    eroded = -_box_dilate(-alpha, radius)
    return ((dilated > background_threshold) & (eroded < foreground_threshold)).to(alpha.dtype)


def _gaussian_derivative_kernels(
    sigma: float, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Separable Gaussian-derivative kernels, matching the standard matting
    gradient error (and `diffusion.utils.matting_metrics`)."""
    key = (float(sigma), device, dtype)
    cached = _GRADIENT_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    epsilon = 1e-2
    half = math.ceil(sigma * math.sqrt(-2.0 * math.log(math.sqrt(2.0 * math.pi) * sigma * epsilon)))
    coords = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    gauss = torch.exp(-(coords**2) / (2.0 * sigma**2)) / (sigma * math.sqrt(2.0 * math.pi))
    dgauss = -coords * gauss / (sigma**2)
    kernel_x = gauss.unsqueeze(1) * dgauss.unsqueeze(0)
    kernel_y = kernel_x.t()
    kernel_x = kernel_x / kernel_x.abs().pow(2).sum().sqrt()
    kernel_y = kernel_y / kernel_y.abs().pow(2).sum().sqrt()
    kernels = (
        kernel_x.to(dtype)[None, None],
        kernel_y.to(dtype)[None, None],
    )
    _GRADIENT_KERNEL_CACHE[key] = kernels
    return kernels


def gradient_amplitude(alpha: torch.Tensor, sigma: float = 1.4) -> torch.Tensor:
    """Gradient magnitude of `[B, 1, H, W]` alpha under a Gaussian derivative."""
    kernel_x, kernel_y = _gaussian_derivative_kernels(sigma, alpha.device, alpha.dtype)
    grad_x = F.conv2d(alpha, kernel_x, padding="same")
    grad_y = F.conv2d(alpha, kernel_y, padding="same")
    # The clamp keeps the sqrt differentiable where both gradients vanish, which
    # is most of the image for a near-binary matte.
    return (grad_x**2 + grad_y**2).clamp_min(1e-12).sqrt()


def band_matting_losses(
    predicted_alpha: torch.Tensor,
    target_alpha: torch.Tensor,
    band: torch.Tensor,
    sigma: float = 1.4,
) -> Dict[str, torch.Tensor]:
    """SAD, MSE, and gradient error restricted to the unknown band.

    All three are band means, so they are comparable to each other and
    independent of image size and band width.
    """
    if predicted_alpha.shape != target_alpha.shape:
        raise ValueError(
            f"Alpha shape mismatch: {tuple(predicted_alpha.shape)} vs {tuple(target_alpha.shape)}"
        )
    # Accumulate in fp32. Under bf16 autocast the squared gradient-amplitude
    # error runs out of mantissa: the amplitudes are small over a near-binary
    # matte, and squaring them lands below bf16's ~3 significant digits.
    predicted_alpha = predicted_alpha.float()
    target_alpha = target_alpha.float()
    band = band.float()
    # An empty band (a fully opaque or fully empty matte) must contribute zero
    # rather than a division by zero.
    count = band.sum().clamp_min(1.0)
    difference = predicted_alpha - target_alpha
    amplitude_error = gradient_amplitude(predicted_alpha, sigma) - gradient_amplitude(
        target_alpha, sigma
    )
    return {
        "sad": (difference.abs() * band).sum() / count,
        "mse": (difference.square() * band).sum() / count,
        "grad": (amplitude_error.square() * band).sum() / count,
    }


def sample_band_radius(
    radius_min: int, radius_max: int, generator: Optional[torch.Generator] = None
) -> int:
    """Draw a band width. Randomizing it stops the model from fitting one
    particular dilation, the way Edit2Perceive randomizes its trimap kernel."""
    radius_min = max(int(radius_min), 0)
    radius_max = max(int(radius_max), radius_min)
    if radius_max == radius_min:
        return radius_min
    span = radius_max - radius_min + 1
    offset = int(torch.randint(0, span, (1,), generator=generator).item())
    return radius_min + offset


def matting_band_loss(
    predicted_images: torch.Tensor,
    target_images: torch.Tensor,
    radius: int,
    sad_weight: float = 1.0,
    mse_weight: float = 1.0,
    grad_weight: float = 1.0,
    sigma: float = 1.4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Weighted band loss for a predicted and target matte in `[-1, 1]`.

    Returns the combined loss and the individual terms for logging.
    """
    predicted_alpha = alpha_from_model_space(predicted_images)
    target_alpha = alpha_from_model_space(target_images)
    with torch.no_grad():
        # The band comes from the ground truth alone, so it carries no gradient
        # and cannot be widened by the model to make its own job easier.
        band = unknown_band(target_alpha.detach().float(), radius).to(target_alpha.dtype)
    terms = band_matting_losses(predicted_alpha, target_alpha, band, sigma=sigma)
    total = (
        float(sad_weight) * terms["sad"]
        + float(mse_weight) * terms["mse"]
        + float(grad_weight) * terms["grad"]
    )
    # Named without a "band" prefix because callers namespace these terms.
    terms["fraction"] = band.mean()
    return total, terms

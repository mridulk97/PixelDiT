from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
T2I_ROOT = REPO_ROOT / "t2i"
for directory in (str(REPO_ROOT), str(T2I_ROOT)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from diffusion.data.datasets.pixdit_datasets import (
    AM2KMattingDataset,
    Distinctions646MattingDataset,
    _group_stratified_subset,
)
from diffusion import Scheduler
from diffusion.model.lora import (
    configure_matting_trainable_parameters,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)
from diffusion.model.trainer import PixDiTTrainer
from diffusion.utils.matting_metrics import compute_matting_metrics
from pixdit_core.pixeldit_t2i import PixDiT_T2I
from diffusion.model.matting_losses import (
    alpha_from_model_space,
    matting_band_loss,
    unknown_band,
)
from train_matting import (
    _band_loss_scale,
    _rotating_preview_indices,
    _subset_manifest,
    _subset_records,
    _fires_on,
    _decode_deterministic,
    _deterministic_inputs,
    _flow_loss,
    _sample_training_grid,
    _validation_losses,
)


def tiny_core(mode="none", rope_mode="aligned"):
    return PixDiT_T2I(
        in_channels=3,
        num_groups=4,
        hidden_size=32,
        pixel_hidden_size=4,
        pixel_attn_hidden_size=32,
        pixel_num_groups=4,
        patch_depth=1,
        pixel_depth=1,
        num_text_blocks=1,
        patch_size=2,
        txt_embed_dim=16,
        txt_max_length=5,
        conditioning_mode=mode,
        sequence_rope_mode=rope_mode,
    )


class TinyWrapper(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.core = tiny_core(mode)

    def forward(self, x, timestep, y, condition_image=None, **_kwargs):
        if y.dim() == 4:
            y = y.squeeze(1)
        return {"x": self.core(x, timestep, y, condition_image=condition_image)}

    def forward_with_dpmsolver(self, x, timestep, y, **kwargs):
        return self.forward(x, timestep, y, **kwargs)["x"]


class ConditioningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(2, 3, 8, 8)
        self.condition = torch.randn_like(self.x)
        self.t = torch.tensor([100.0, 200.0])
        self.y = torch.randn(2, 5, 16)

    def test_all_modes_forward_shapes_and_input_widths(self):
        expected = {
            "none": (12, 3),
            "patch": (24, 3),
            "pixel": (12, 6),
            "both": (24, 6),
            "sequence": (12, 3),
            "sequence_pixel": (12, 6),
        }
        for mode, widths in expected.items():
            with self.subTest(mode=mode):
                model = tiny_core(mode)
                condition = None if mode == "none" else self.condition
                output = model(self.x, self.t, self.y, condition_image=condition)
                self.assertEqual(tuple(output.shape), tuple(self.x.shape))
                self.assertEqual(model.s_embedder.proj.in_features, widths[0])
                self.assertEqual(model.pixel_embedder.proj.in_features, widths[1])

    def test_condition_shape_validation(self):
        model = tiny_core("both")
        with self.assertRaisesRegex(ValueError, "must match"):
            model(self.x, self.t, self.y, condition_image=self.condition[:, :, :-1])

    def test_sequence_uses_2l_then_slices_to_l(self):
        model = tiny_core("sequence")
        observed = {}

        def patch_hook(_module, inputs):
            observed["patch_length"] = inputs[0].shape[1]

        def pixel_hook(_module, inputs):
            observed["pixel_condition_rows"] = inputs[1].shape[0]

        handles = [
            model.patch_blocks[0].register_forward_pre_hook(patch_hook),
            model.pixel_blocks[0].register_forward_pre_hook(pixel_hook),
        ]
        try:
            model(self.x, self.t, self.y, condition_image=self.condition)
        finally:
            for handle in handles:
                handle.remove()
        target_length = (8 // 2) ** 2
        self.assertEqual(observed["patch_length"], 2 * target_length)
        self.assertEqual(observed["pixel_condition_rows"], 2 * target_length)

    def test_sequence_rope_aligned_and_offset(self):
        aligned = tiny_core("sequence", "aligned")._sequence_positions(4, 4, torch.device("cpu"))
        offset = tiny_core("sequence", "offset")._sequence_positions(4, 4, torch.device("cpu"))
        self.assertTrue(torch.equal(aligned[:16], aligned[16:]))
        self.assertFalse(torch.equal(offset[:16], offset[16:]))

        native = tiny_core("sequence", "aligned")._sequence_positions(64, 64, torch.device("cpu"))
        self.assertEqual(native.shape[0], 8192)


class InitializationTests(unittest.TestCase):
    @staticmethod
    def _trainer(mode, proj_init="zero"):
        config = SimpleNamespace(
            model=SimpleNamespace(
                conditioning_mode=mode,
                conditioning_proj_init=proj_init,
                sequence_rope_mode="aligned",
                sequence_rope_offset=None,
                use_sequence_type_embedding=True,
            )
        )
        return PixDiTTrainer(
            image_size=8,
            caption_channels=16,
            model_max_length=5,
            config=config,
            extra={
                "patch_size": 2,
                "num_groups": 4,
                "hidden_size": 32,
                "pixel_hidden_size": 4,
                "pixel_attn_hidden_size": 32,
                "pixel_num_groups": 4,
                "patch_depth": 1,
                "pixel_depth": 1,
                "txt_embed_dim": 16,
                "txt_max_length": 5,
            },
        )

    def test_channel_concat_zero_init_preserves_pretrained_projection(self):
        torch.manual_seed(11)
        base = self._trainer("none")
        state = base.state_dict()
        widened = self._trainer("both")
        result = widened.load_state_dict(state, strict=False)
        self.assertEqual(result.unexpected_keys, [])
        for key in ("core.s_embedder.proj.weight", "core.pixel_embedder.proj.weight"):
            old = state[key]
            new = widened.state_dict()[key]
            self.assertTrue(torch.equal(new[:, : old.shape[1]], old))
            self.assertTrue(torch.count_nonzero(new[:, old.shape[1] :]) == 0)
        self.assertTrue(
            torch.equal(widened.core.s_embedder.proj.bias, state["core.s_embedder.proj.bias"])
        )

    def test_zero_init_widening_matches_the_unconditioned_projection(self):
        """The widened layer must compute exactly Wx at initialization."""
        torch.manual_seed(12)
        base = self._trainer("none")
        state = base.state_dict()
        widened = self._trainer("both")
        widened.load_state_dict(state, strict=False)
        patches = torch.randn(2, 4, 12)
        condition = torch.randn(2, 4, 12)
        reference = base.core.s_embedder(patches)
        conditioned = widened.core.s_embedder(torch.cat([patches, condition], dim=-1))
        self.assertTrue(torch.allclose(reference, conditioned, atol=1e-6))

    def test_balanced_init_reproduces_the_legacy_sqrt2_expansion(self):
        torch.manual_seed(11)
        base = self._trainer("none")
        state = base.state_dict()
        widened = self._trainer("both", proj_init="balanced")
        widened.load_state_dict(state, strict=False)
        for key in ("core.s_embedder.proj.weight", "core.pixel_embedder.proj.weight"):
            old = state[key]
            new = widened.state_dict()[key]
            self.assertTrue(torch.equal(new[:, : old.shape[1]], old / np.sqrt(2.0)))
            self.assertTrue(torch.equal(new[:, old.shape[1] :], old / np.sqrt(2.0)))

    def test_sequence_projection_is_unchanged_and_shared(self):
        torch.manual_seed(13)
        base = self._trainer("none")
        state = base.state_dict()
        sequence = self._trainer("sequence")
        result = sequence.load_state_dict(state, strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(
            sorted(result.missing_keys),
            ["core.reference_type_embedding", "core.target_type_embedding"],
        )
        self.assertTrue(torch.equal(sequence.core.s_embedder.proj.weight, state["core.s_embedder.proj.weight"]))
        self.assertEqual(sequence.core.s_embedder.proj.in_features, 12)

    def test_sequence_type_embeddings_break_the_stream_symmetry(self):
        """Aligned RoPE leaves these as the only target/reference distinction."""
        for mode in ("sequence", "sequence_pixel"):
            with self.subTest(mode=mode):
                core = self._trainer(mode).core
                self.assertIsNotNone(core.target_type_embedding)
                self.assertIsNotNone(core.reference_type_embedding)
                self.assertTrue(torch.count_nonzero(core.target_type_embedding) > 0)
                self.assertTrue(torch.count_nonzero(core.reference_type_embedding) > 0)
                self.assertFalse(
                    torch.equal(core.target_type_embedding, core.reference_type_embedding)
                )

    def test_sequence_pixel_widens_only_the_pixel_projection(self):
        torch.manual_seed(15)
        base = self._trainer("none")
        state = base.state_dict()
        model = self._trainer("sequence_pixel")
        model.load_state_dict(state, strict=False)
        self.assertEqual(model.core.s_embedder.proj.in_features, 12)
        self.assertEqual(model.core.pixel_embedder.proj.in_features, 6)
        old = state["core.pixel_embedder.proj.weight"]
        new = model.state_dict()["core.pixel_embedder.proj.weight"]
        self.assertTrue(torch.equal(new[:, : old.shape[1]], old))
        self.assertTrue(torch.count_nonzero(new[:, old.shape[1] :]) == 0)


class DeterministicFlowTests(unittest.TestCase):
    """The deterministic regime must give the model no view of the target."""

    @staticmethod
    def _config(deterministic):
        return SimpleNamespace(
            scheduler=SimpleNamespace(
                deterministic_flow=deterministic,
                train_sampling_steps=1000,
            )
        )

    def test_model_input_carries_no_trace_of_the_target(self):
        target = torch.randn(2, 3, 4, 4)
        x_t, timesteps = _deterministic_inputs(target, 1000)
        self.assertEqual(x_t.shape, target.shape)
        # Exactly zero, not merely small: a scaled copy of the target is still
        # the target, and normalization layers can undo an attenuation.
        self.assertEqual(torch.count_nonzero(x_t).item(), 0)
        self.assertTrue(torch.equal(timesteps, torch.full((2,), 999, dtype=torch.long)))

    def test_loss_is_the_single_step_regression_to_minus_target(self):
        torch.manual_seed(3)
        model = TinyWrapper("both")
        nn.init.normal_(model.core.final_layer.linear.weight, std=0.02)
        target = torch.randn(2, 3, 4, 4)
        kwargs = {
            "y": torch.randn(2, 5, 16),
            "condition_image": torch.randn(2, 3, 4, 4),
        }
        loss, _ = _flow_loss(model, None, target, kwargs, self._config(True))
        x_t, timesteps = _deterministic_inputs(target, 1000)
        with torch.no_grad():
            expected_output = model(x_t, timesteps, **kwargs)["x"]
        expected = ((expected_output.float() + target.float()) ** 2).mean()
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

    def test_decode_inverts_the_training_target(self):
        """A model that predicts the target exactly must decode to it."""
        class NegateTarget(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, x, timestep, **_kwargs):
                return {"x": -self.value}

        target = torch.randn(2, 3, 4, 4)
        decoded = _decode_deterministic(NegateTarget(target), target, {}, 1000)
        self.assertTrue(torch.equal(decoded, target))

    def test_loss_falls_back_to_the_stochastic_path(self):
        torch.manual_seed(5)
        model = TinyWrapper("both")
        nn.init.normal_(model.core.final_layer.linear.weight, std=0.02)
        flow_matching = Scheduler(
            "1000",
            noise_schedule="linear_flow",
            predict_flow_v=True,
            learn_sigma=False,
            pred_sigma=False,
            flow_shift=4.0,
        )
        target = torch.randn(1, 3, 4, 4)
        kwargs = {
            "y": torch.randn(1, 5, 16),
            "condition_image": torch.randn(1, 3, 4, 4),
        }
        timesteps = torch.tensor([250], dtype=torch.long)
        noise = torch.randn_like(target)
        loss, _ = _flow_loss(
            model, flow_matching, target, kwargs, self._config(False),
            timesteps=timesteps, noise=noise,
        )
        expected = flow_matching.training_losses(
            model, target, timesteps, noise=noise, model_kwargs=kwargs
        )["loss"].mean()
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

    def test_deterministic_decode_depends_on_the_condition(self):
        """With a zero input the condition is the only source of information."""
        torch.manual_seed(7)
        for mode in ("both", "sequence", "sequence_pixel"):
            with self.subTest(mode=mode):
                model = TinyWrapper(mode).eval()
                nn.init.normal_(model.core.final_layer.linear.weight, std=0.02)
                target = torch.randn(1, 3, 4, 4)
                y = torch.randn(1, 5, 16)
                first = _decode_deterministic(
                    model, target, {"y": y, "condition_image": torch.randn(1, 3, 4, 4)}, 1000
                )
                second = _decode_deterministic(
                    model, target, {"y": y, "condition_image": torch.randn(1, 3, 4, 4)}, 1000
                )
                self.assertTrue((first - second).abs().max().item() > 1e-8)


class LoRATests(unittest.TestCase):
    def test_generated_triplet_grid(self):
        for deterministic in (False, True):
            with self.subTest(deterministic_flow=deterministic):
                model = TinyWrapper("both")
                target = torch.randn(1, 3, 4, 4).clamp(-1, 1)
                condition = torch.randn_like(target).clamp(-1, 1)
                batch = (target, None, None, None, None, None, None, None, condition)
                grid, generated_mse = _sample_training_grid(
                    model,
                    batch,
                    torch.randn(1, 5, 16),
                    torch.ones(1, 5),
                    sample_steps=2,
                    flow_shift=4.0,
                    num_examples=1,
                    config=DeterministicFlowTests._config(deterministic),
                )
                self.assertEqual(grid.shape[0], 3)
                self.assertGreater(grid.shape[2], 3 * target.shape[-1])
                self.assertTrue(np.isfinite(generated_mse))

    def test_validation_is_condition_shuffled_at_batch_one_memory(self):
        model = TinyWrapper("both")
        model.train()
        target = torch.randn(2, 3, 4, 4).clamp(-1, 1)
        condition = torch.randn_like(target).clamp(-1, 1)
        batch = (target, None, None, None, None, None, None, None, condition)
        flow_matching = Scheduler(
            "1000",
            noise_schedule="linear_flow",
            predict_flow_v=True,
            learn_sigma=False,
            pred_sigma=False,
            flow_shift=4.0,
        )
        for deterministic in (False, True):
            with self.subTest(deterministic_flow=deterministic):
                model.train()
                correct, shuffled = _validation_losses(
                    model,
                    flow_matching,
                    batch,
                    torch.randn(1, 5, 16),
                    torch.ones(1, 5),
                    DeterministicFlowTests._config(deterministic),
                )
                self.assertTrue(torch.isfinite(correct))
                self.assertTrue(torch.isfinite(shuffled))
                self.assertTrue(model.training)

    def test_small_forward_backward_for_channel_and_sequence_paths(self):
        x = torch.randn(1, 3, 4, 4)
        condition = torch.randn_like(x)
        timestep = torch.tensor([250], dtype=torch.long)
        text = torch.randn(1, 5, 16)
        flow_matching = Scheduler(
            "1000",
            noise_schedule="linear_flow",
            predict_flow_v=True,
            learn_sigma=False,
            pred_sigma=False,
            flow_shift=4.0,
        )
        for mode in ("patch", "pixel", "both", "sequence", "sequence_pixel"):
            with self.subTest(mode=mode):
                model = TinyWrapper(mode)
                configure_matting_trainable_parameters(model, rank=2, alpha=2, dropout=0)
                model.core.grad_checkpointing = mode.startswith("sequence")
                nn.init.normal_(model.core.final_layer.linear.weight, std=0.02)
                loss = flow_matching.training_losses(
                    model,
                    x,
                    timestep,
                    model_kwargs={"y": text, "condition_image": condition},
                )["loss"].mean()
                loss.backward()
                gradients = [
                    parameter.grad
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and name.endswith("lora_B.weight")
                ]
                self.assertTrue(
                    any(gradient is not None and torch.any(gradient != 0) for gradient in gradients)
                )

    def test_frozen_and_trainable_input_projections_per_mode(self):
        expectations = {
            "patch": {"s_embedder": True, "pixel_embedder": False, "type_embeddings": False},
            "pixel": {"s_embedder": False, "pixel_embedder": True, "type_embeddings": False},
            "both": {"s_embedder": True, "pixel_embedder": True, "type_embeddings": False},
            "sequence": {"s_embedder": False, "pixel_embedder": False, "type_embeddings": True},
            "sequence_pixel": {"s_embedder": False, "pixel_embedder": True, "type_embeddings": True},
        }
        for mode, expected in expectations.items():
            with self.subTest(mode=mode):
                model = TinyWrapper(mode)
                configure_matting_trainable_parameters(model, rank=2, alpha=2, dropout=0)
                trainable = {name for name, value in model.named_parameters() if value.requires_grad}
                self.assertEqual(
                    "core.s_embedder.proj.weight" in trainable, expected["s_embedder"]
                )
                self.assertEqual(
                    "core.pixel_embedder.proj.weight" in trainable, expected["pixel_embedder"]
                )
                for name in ("core.target_type_embedding", "core.reference_type_embedding"):
                    self.assertEqual(name in trainable, expected["type_embeddings"], name)

    def test_trainable_modules_and_adapter_reload(self):
        torch.manual_seed(17)
        model = TinyWrapper("both")
        base_state = model.state_dict()
        info = configure_matting_trainable_parameters(model, rank=2, alpha=2, dropout=0)
        self.assertEqual(len(info["target_modules"]), 13)
        trainable_names = {name for name, value in model.named_parameters() if value.requires_grad}
        self.assertIn("core.s_embedder.proj.weight", trainable_names)
        self.assertIn("core.pixel_embedder.proj.weight", trainable_names)
        self.assertIn("core.final_layer.linear.weight", trainable_names)
        self.assertTrue(any(name.endswith("lora_A.weight") for name in trainable_names))
        self.assertFalse(any("qkv_y" in name for name in trainable_names))

        with torch.no_grad():
            for name, value in model.named_parameters():
                if value.requires_grad:
                    value.add_(torch.randn_like(value) * 0.01)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "adapter.pth")
            save_adapter_checkpoint(path, model, info, step=9, epoch=2)
            clone = TinyWrapper("both")
            clone.load_state_dict(base_state)
            configure_matting_trainable_parameters(clone, rank=2, alpha=2, dropout=0)
            payload = load_adapter_checkpoint(path, clone)
            self.assertEqual(payload["step"], 9)
            for (name, value), (clone_name, clone_value) in zip(
                model.named_parameters(), clone.named_parameters()
            ):
                self.assertEqual(name, clone_name)
                if value.requires_grad:
                    self.assertTrue(torch.equal(value, clone_value), name)
            model.eval()
            clone.eval()
            inputs = (
                torch.randn(1, 3, 4, 4),
                torch.tensor([400.0]),
                torch.randn(1, 5, 16),
                torch.randn(1, 3, 4, 4),
            )
            first = model.core(inputs[0], inputs[1], inputs[2], condition_image=inputs[3])
            restored = clone.core(inputs[0], inputs[1], inputs[2], condition_image=inputs[3])
            self.assertTrue(torch.equal(first, restored))


class BandLossTests(unittest.TestCase):
    """Trimap-band losses: the term that targets where the error actually is."""

    @staticmethod
    def _disc(size=64, radius=20, soft=0):
        """A soft-edged disc as a stand-in for an alpha matte."""
        coords = torch.arange(size, dtype=torch.float32) - size / 2
        distance = (coords[:, None] ** 2 + coords[None, :] ** 2).sqrt()
        if soft:
            alpha = ((radius + soft - distance) / (2 * soft)).clamp(0, 1)
        else:
            alpha = (distance <= radius).float()
        return alpha[None, None]

    def test_band_hugs_the_boundary_and_excludes_flat_regions(self):
        alpha = self._disc()
        band = unknown_band(alpha, radius=5)
        self.assertEqual(band.shape, alpha.shape)
        # Centre and far corner are unambiguous, so neither may be in the band.
        self.assertEqual(band[0, 0, 32, 32].item(), 0.0)
        self.assertEqual(band[0, 0, 0, 0].item(), 0.0)
        self.assertGreater(band.mean().item(), 0.0)
        # A wider dilation must cover strictly more.
        self.assertGreater(unknown_band(alpha, 10).mean(), band.mean())

    def test_uniform_matte_has_an_empty_band_and_no_nan(self):
        for value in (0.0, 1.0):
            with self.subTest(value=value):
                images = torch.full((1, 3, 32, 32), value * 2 - 1)
                self.assertEqual(unknown_band(alpha_from_model_space(images), 4).sum().item(), 0.0)
                loss, _ = matting_band_loss(images, images, radius=4)
                self.assertEqual(loss.item(), 0.0)
                self.assertFalse(torch.isnan(loss))

    def test_perfect_prediction_is_exactly_zero(self):
        target = self._disc(soft=3).repeat(1, 3, 1, 1) * 2 - 1
        loss, terms = matting_band_loss(target, target, radius=8)
        self.assertEqual(loss.item(), 0.0)
        for name in ("sad", "mse", "grad"):
            self.assertEqual(terms[name].item(), 0.0, name)

    def test_worse_prediction_scores_higher(self):
        target = self._disc(soft=3).repeat(1, 3, 1, 1) * 2 - 1
        close = (target + 0.05 * torch.randn_like(target)).clamp(-1, 1)
        far = self._disc(radius=14, soft=3).repeat(1, 3, 1, 1) * 2 - 1
        self.assertLess(
            matting_band_loss(close, target, radius=8)[0].item(),
            matting_band_loss(far, target, radius=8)[0].item(),
        )

    def test_gradient_reaches_the_prediction_but_not_the_band(self):
        target = self._disc(soft=3).repeat(1, 3, 1, 1) * 2 - 1
        prediction = (target + 0.1).clone().requires_grad_(True)
        loss, _ = matting_band_loss(prediction, target, radius=8)
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.any(prediction.grad != 0))
        # The band is derived from the target under no_grad, so a model cannot
        # widen it to make its own job easier.
        self.assertFalse(unknown_band(alpha_from_model_space(target), 8).requires_grad)

    def test_terms_are_resolution_independent(self):
        """Band means, unlike a fixed /1000, must not move with image size."""
        small = self._disc(size=64, radius=20, soft=3)
        large = torch.nn.functional.interpolate(
            small, scale_factor=2, mode="bilinear", align_corners=False
        )
        def score(alpha, radius):
            target = alpha.repeat(1, 3, 1, 1) * 2 - 1
            prediction = (target * 0.8).clamp(-1, 1)
            return matting_band_loss(prediction, target, radius=radius)[0].item()
        # Radius scales with the image so the band covers the same structure.
        self.assertAlmostEqual(score(small, 5), score(large, 10), delta=0.02)

    def test_periodic_cadences_are_independent(self):
        """Preview and checkpoint intervals must not gate each other.

        They used to: the preview block lived inside the checkpoint branch, so
        wandb_image_interval=50 with adapter_save_steps=100 produced previews
        every 100 steps and the finer setting did nothing.
        """
        save_every, image_every, max_steps = 100, 50, 1000
        previews = [s for s in range(1, max_steps + 1) if _fires_on(s, image_every, max_steps)]
        saves = [s for s in range(1, max_steps + 1) if _fires_on(s, save_every, max_steps)]
        self.assertEqual(previews[:4], [50, 100, 150, 200])
        self.assertEqual(saves[:4], [100, 200, 300, 400])
        self.assertEqual(len(previews), 2 * len(saves))

    def test_fires_on_always_includes_the_final_step(self):
        self.assertTrue(_fires_on(777, 100, 777))
        self.assertFalse(_fires_on(777, 100, 1000))
        self.assertTrue(_fires_on(300, 100, 1000))
        # A non-positive interval must clamp rather than divide by zero.
        self.assertTrue(_fires_on(7, 0, 1000))

    def test_warmup_ramps_the_band_weight(self):
        config = SimpleNamespace(
            train=SimpleNamespace(matting_band_loss_weight=2.0, matting_band_warmup_steps=100)
        )
        self.assertEqual(_band_loss_scale(config, 0), 0.0)
        self.assertAlmostEqual(_band_loss_scale(config, 50), 1.0)
        self.assertEqual(_band_loss_scale(config, 100), 2.0)
        self.assertEqual(_band_loss_scale(config, 500), 2.0)
        config.train.matting_band_warmup_steps = 0
        self.assertEqual(_band_loss_scale(config, 0), 2.0)
        config.train.matting_band_loss_weight = 0.0
        self.assertEqual(_band_loss_scale(config, 500), 0.0)

    def test_flow_loss_adds_the_band_term_and_reports_components(self):
        torch.manual_seed(11)
        model = TinyWrapper("both")
        nn.init.normal_(model.core.final_layer.linear.weight, std=0.02)
        target = torch.randn(1, 3, 8, 8).clamp(-1, 1)
        kwargs = {"y": torch.randn(1, 5, 16), "condition_image": torch.randn(1, 3, 8, 8)}
        config = SimpleNamespace(
            scheduler=SimpleNamespace(deterministic_flow=True, train_sampling_steps=1000),
            train=SimpleNamespace(
                matting_band_sad_weight=1.0,
                matting_band_mse_weight=1.0,
                matting_band_grad_weight=1.0,
                matting_band_radius_min=2,
                matting_band_radius_max=2,
            ),
        )
        base, base_parts = _flow_loss(model, None, target, kwargs, config, band_scale=0.0)
        total, parts = _flow_loss(model, None, target, kwargs, config, band_scale=1.0)
        self.assertEqual(set(base_parts), {"base"})
        for name in ("base", "band_sad", "band_mse", "band_grad", "band_total", "band_fraction"):
            self.assertIn(name, parts)
        self.assertAlmostEqual(base.item(), parts["base"].item(), places=5)
        self.assertGreater(total.item(), base.item())

    def test_band_term_requires_deterministic_flow(self):
        config = SimpleNamespace(
            scheduler=SimpleNamespace(deterministic_flow=False, train_sampling_steps=1000),
            train=SimpleNamespace(),
        )
        with self.assertRaisesRegex(ValueError, "deterministic_flow"):
            _flow_loss(TinyWrapper("both"), None, torch.randn(1, 3, 4, 4), {}, config, band_scale=1.0)


class DataAndMetricTests(unittest.TestCase):
    def test_am2k_pairing_preprocessing_and_subset(self):
        root = Path("/scratch/mridul/data/matting/am-2k")
        if not root.is_dir():
            self.skipTest("AM-2K is not available on this machine")
        dataset = AM2KMattingDataset(
            data_dir=str(root),
            resolution=32,
            max_length=5,
            extra={"split": "train", "overfit_samples": 16, "overfit_seed": 2025},
        )
        self.assertEqual(dataset.full_dataset_size, 1800)
        self.assertEqual(len(dataset), 16)
        self.assertEqual(len({record["category"] for record in dataset.dataset}), 16)
        target, prompt, mask, data_info, *tail = dataset[0]
        condition = tail[-1]
        self.assertEqual(tuple(target.shape), (3, 32, 32))
        self.assertEqual(tuple(condition.shape), (3, 32, 32))
        self.assertTrue(torch.equal(target[0], target[1]))
        self.assertGreaterEqual(float(target.min()), -1.0)
        self.assertLessEqual(float(target.max()), 1.0)
        self.assertEqual(mask.shape[-1], 5)
        self.assertEqual(data_info["sample_id"], dataset.dataset[0]["sample_id"])
        self.assertTrue(prompt.startswith("Transform to matting map"))

    def test_band_metrics_isolate_the_boundary(self):
        """Whole-image metrics stay flattering while band metrics do not."""
        from diffusion.utils.matting_metrics import compute_matting_metrics

        size = 64
        coords = np.arange(size, dtype=np.float32) - size / 2
        distance = np.sqrt(coords[:, None] ** 2 + coords[None, :] ** 2)
        target = (distance <= 20).astype(np.float32)
        # Error confined to a ring around the boundary: a blurred edge.
        prediction = np.clip((24 - distance) / 8.0, 0.0, 1.0).astype(np.float32)
        values = compute_matting_metrics(prediction, target, band_radius=6)
        self.assertLess(values["band_fraction"], 0.5)
        self.assertGreater(values["band_error_share"], 0.9)
        self.assertGreater(values["band_mse"], values["mse"])
        exact = compute_matting_metrics(target, target, band_radius=6)
        self.assertEqual(exact["band_mse"], 0.0)
        self.assertEqual(exact["band_sad"], 0.0)

    def test_matting_metrics(self):
        target = np.zeros((12, 12), dtype=np.float32)
        target[3:9, 3:9] = 1.0
        perfect = compute_matting_metrics(target, target)
        descriptors = {"band_fraction", "band_error_share"}
        self.assertTrue(
            all(value == 0.0 for name, value in perfect.items() if name not in descriptors),
            perfect,
        )
        # The band itself is a property of the target, so it is non-empty even
        # when the prediction is exact.
        self.assertGreater(perfect["band_fraction"], 0.0)
        wrong = compute_matting_metrics(np.zeros_like(target), target)
        self.assertGreater(wrong["sad"], 0.0)
        self.assertGreater(wrong["mse"], 0.0)
        self.assertGreater(wrong["mad"], 0.0)
        self.assertGreater(wrong["gradient"], 0.0)
        self.assertGreater(wrong["connectivity"], 0.0)


class RefinementHeadTests(unittest.TestCase):
    """The head is the only module that couples pixels across a patch seam."""

    @staticmethod
    def _core(mode="both", **head_kwargs):
        return PixDiT_T2I(
            in_channels=3,
            num_groups=4,
            hidden_size=32,
            pixel_hidden_size=4,
            pixel_attn_hidden_size=32,
            pixel_num_groups=4,
            patch_depth=1,
            pixel_depth=1,
            num_text_blocks=1,
            patch_size=2,
            txt_embed_dim=16,
            txt_max_length=5,
            conditioning_mode=mode,
            use_refine_head=True,
            **head_kwargs,
        )

    @staticmethod
    def _inputs(batch=2, size=8):
        return (
            torch.randn(batch, 3, size, size),
            torch.full((batch,), 999.0),
            torch.randn(batch, 5, 16),
            torch.randn(batch, 3, size, size),
        )

    def test_head_is_identity_at_initialization(self):
        torch.manual_seed(3)
        plain = tiny_core("both")
        torch.manual_seed(3)
        refined = self._core("both")
        missing, unexpected = refined.load_state_dict(plain.state_dict(), strict=False)
        self.assertEqual(unexpected, [])
        self.assertTrue(all(key.startswith("refine_head.") for key in missing))
        self.assertTrue(missing, "expected the head to be the only new parameters")

        x, t, y, condition = self._inputs()
        with torch.no_grad():
            self.assertTrue(
                torch.equal(
                    plain(x, t, y, condition_image=condition),
                    refined(x, t, y, condition_image=condition),
                )
            )

    def test_receptive_field_spans_a_patch_seam(self):
        core = self._core("both")
        head = core.refine_head
        # A pixel must reach across at least one seam in every direction, so
        # the field has to exceed patch_size rather than merely match it.
        self.assertGreater(head.receptive_field, core.patch_size)

        for parameter in head.parameters():
            nn.init.constant_(parameter, 0.05)
        probe = torch.zeros(1, head.in_channels, 41, 41, requires_grad=True)
        head(probe)[0, :, 20, 20].sum().backward()
        touched = (probe.grad.abs().sum(dim=(0, 1)) > 0).any(dim=1).nonzero().flatten()
        measured = int(touched.max() - touched.min()) + 1
        self.assertEqual(measured, head.receptive_field)

    def test_zero_init_gates_gradient_like_a_lora_branch(self):
        torch.manual_seed(5)
        core = self._core("both")
        # A freshly built core zero-inits final_layer; the pretrained checkpoint
        # does not, and an identically zero output would mask the gradient.
        nn.init.normal_(core.final_layer.linear.weight, std=0.02)
        head = core.refine_head
        x, t, y, condition = self._inputs()
        target = torch.randn_like(x)

        ((core(x, t, y, condition_image=condition) - target) ** 2).mean().backward()
        with_gradient = {
            name for name, p in head.named_parameters() if p.grad.abs().sum() > 0
        }
        self.assertEqual(with_gradient, {"body.8.weight", "body.8.bias"})

        torch.optim.SGD(head.parameters(), lr=1.0).step()
        head.zero_grad()
        ((core(x, t, y, condition_image=condition) - target) ** 2).mean().backward()
        self.assertTrue(all(p.grad.abs().sum() > 0 for p in head.parameters()))

    def test_head_trains_in_full_and_survives_the_adapter_round_trip(self):
        config = SimpleNamespace(
            model=SimpleNamespace(
                conditioning_mode="both",
                conditioning_proj_init="zero",
                sequence_rope_mode="aligned",
                sequence_rope_offset=None,
                use_sequence_type_embedding=True,
                use_refine_head=True,
                refine_head_width=8,
                refine_head_dilations=[1, 2],
            )
        )
        model = PixDiTTrainer(
            image_size=8,
            caption_channels=16,
            model_max_length=5,
            config=config,
            extra={
                "patch_size": 2,
                "num_groups": 4,
                "hidden_size": 32,
                "pixel_hidden_size": 4,
                "pixel_attn_hidden_size": 32,
                "pixel_num_groups": 4,
                "patch_depth": 1,
                "pixel_depth": 1,
                "txt_embed_dim": 16,
                "txt_max_length": 5,
            },
        )
        info = configure_matting_trainable_parameters(model, rank=2, alpha=2.0)
        self.assertIsNotNone(info["refine_head"])
        self.assertEqual(info["refine_head"]["dilations"], (1, 2))

        head_parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if "refine_head" in name
        }
        self.assertTrue(head_parameters)
        self.assertTrue(all(p.requires_grad for p in head_parameters.values()))
        # Convolutions must not be LoRA-wrapped; the head has nothing to adapt.
        self.assertFalse(any("refine_head" in name for name in info["target_modules"]))

        with torch.no_grad():
            for parameter in head_parameters.values():
                parameter.add_(torch.randn_like(parameter) * 0.1)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "adapter.pth")
            save_adapter_checkpoint(path, model, info, step=1)
            saved = torch.load(path, map_location="cpu")["adapter_state_dict"]
            self.assertEqual(
                {name for name in saved if "refine_head" in name},
                set(head_parameters),
            )
            for parameter in head_parameters.values():
                with torch.no_grad():
                    parameter.zero_()
            load_adapter_checkpoint(path, model)
        for name, parameter in head_parameters.items():
            self.assertTrue(torch.allclose(parameter, saved[name]), name)

    def test_unconditioned_mode_builds_without_a_guide(self):
        core = self._core("none")
        self.assertEqual(core.refine_head.guide_channels, 0)
        self.assertEqual(
            core.refine_head.in_channels,
            core.pixel_hidden_size + core.out_channels,
        )
        x, t, y, _ = self._inputs()
        with torch.no_grad():
            self.assertEqual(core(x, t, y).shape, x.shape)

    def test_invalid_dilations_are_rejected(self):
        with self.assertRaises(ValueError):
            self._core("both", refine_head_dilations=())
        with self.assertRaises(ValueError):
            self._core("both", refine_head_dilations=(1, 0))


class Distinctions646Tests(unittest.TestCase):
    """D-646 is where partial coverage actually lives, so the loader has to be
    right before any transparency claim can be."""

    @staticmethod
    def _build_tree(root: Path, foregrounds=4, backgrounds=8, size=8):
        """A miniature D-646: FG/GT pairs plus a background list, as shipped."""
        rng = np.random.RandomState(0)
        for split, count in (("Train", foregrounds), ("Test", 2)):
            for kind in ("FG", "GT"):
                (root / split / kind).mkdir(parents=True, exist_ok=True)
            names = [f"fg{i:03d}.png" for i in range(count)]
            for name in names:
                Image.fromarray(rng.randint(0, 255, (size, size, 3), dtype=np.uint8)).save(
                    root / split / "FG" / name
                )
                Image.fromarray(rng.randint(0, 255, (size, size), dtype=np.uint8), mode="L").save(
                    root / split / "GT" / name
                )
            listing = root / split / ("bg_train.txt" if split == "Train" else "bg_test.txt")
            per = backgrounds if split == "Train" else 2
            # Shipped lists are CRLF and end without a trailing newline.
            listing.write_bytes(
                "\r\n".join(f"bg{i:03d}.jpg" for i in range(count * per)).encode()
            )
        backgrounds_dir = root / "backgrounds"
        backgrounds_dir.mkdir(exist_ok=True)
        for i in range(foregrounds * backgrounds):
            Image.fromarray(rng.randint(0, 255, (size // 2, size // 2, 3), dtype=np.uint8)).save(
                backgrounds_dir / f"bg{i:03d}.jpg"
            )
        return root

    def _dataset(self, root, **extra):
        options = {"split": "train", "background_dir": str(Path(root) / "backgrounds")}
        options.update(extra)
        return Distinctions646MattingDataset(
            data_dir=[str(root)], resolution=8, max_length=5, extra=options
        )

    def test_composites_have_the_am2k_item_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._build_tree(Path(directory))
            dataset = self._dataset(root)
            self.assertEqual(dataset.foreground_count, 4)
            self.assertEqual(dataset.num_backgrounds, 8)
            self.assertEqual(len(dataset), 32)
            item = dataset[0]
            self.assertEqual(len(item), 9)
            alpha_rgb, _p, mask, info, _i, _k, sample_id, category, condition = item
            self.assertEqual(alpha_rgb.shape, (3, 8, 8))
            self.assertEqual(condition.shape, (3, 8, 8))
            self.assertEqual(mask.shape, (1, 1, 5))
            self.assertEqual(info["sample_id"], sample_id)
            self.assertEqual(category, Distinctions646MattingDataset.foreground_key(sample_id))
            for tensor in (alpha_rgb, condition):
                self.assertGreaterEqual(tensor.min().item(), -1.0)
                self.assertLessEqual(tensor.max().item(), 1.0)

    def test_composite_matches_the_alpha_blend(self):
        """alpha * fg + (1 - alpha) * bg, exactly as gen_train.py does it."""
        with tempfile.TemporaryDirectory() as directory:
            root = self._build_tree(Path(directory))
            dataset = self._dataset(root)
            record = dataset.dataset[0]
            image, alpha = dataset.composite(record)
            foreground = np.asarray(Image.open(record["foreground_path"]).convert("RGB"), np.float32)
            background = Image.open(dataset._background_path(record["background"])).convert("RGB")
            width, height = foreground.shape[1], foreground.shape[0]
            ratio = max(width / background.size[0], height / background.size[1])
            if ratio > 1:
                background = background.resize(
                    (int(np.ceil(background.size[0] * ratio)), int(np.ceil(background.size[1] * ratio))),
                    Image.Resampling.BICUBIC,
                )
            background = np.asarray(background.crop((0, 0, width, height)), np.float32)
            expected = alpha[..., None] * foreground + (1 - alpha[..., None]) * background
            self.assertTrue(np.allclose(np.asarray(image, np.float32), np.clip(expected, 0, 255).astype(np.uint8), atol=1))

    def test_background_assignment_is_positional_and_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._build_tree(Path(directory))
            dataset = self._dataset(root)
            # gen_train.py walks the list in order: foreground i takes
            # backgrounds [i * num_backgrounds, (i + 1) * num_backgrounds).
            self.assertEqual(dataset.dataset[0]["background"], "bg000.jpg")
            self.assertEqual(dataset.dataset[1]["background"], "bg001.jpg")
            self.assertEqual(dataset.dataset[8]["background"], "bg008.jpg")
            # An overfit subset is meaningless if a sample changes between epochs.
            again = self._dataset(root)
            self.assertTrue(torch.equal(dataset[3][0], again[3][0]))
            self.assertTrue(torch.equal(dataset[3][8], again[3][8]))

    def test_crlf_background_list_without_trailing_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._build_tree(Path(directory))
            dataset = self._dataset(root)
            self.assertEqual(len(dataset.background_names), 32)
            self.assertTrue(all("\r" not in name for name in dataset.background_names))

    def test_foreground_key_groups_composites(self):
        key = Distinctions646MattingDataset.foreground_key
        self.assertEqual(key("004c8d27c4063952a98616dd3c8ab316_0"), "004c8d27c4063952a98616dd3c8ab316")
        self.assertEqual(key("h_29_9"), "h_29")
        self.assertEqual(key("13(2)_57"), "13(2)")

    def test_overfit_subset_spreads_across_foregrounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._build_tree(Path(directory))
            dataset = self._dataset(root, overfit_samples=4, overfit_seed=2025)
            self.assertEqual(len(dataset), 4)
            self.assertEqual(dataset.full_dataset_size, 32)
            self.assertEqual(len({record["category"] for record in dataset.dataset}), 4)
            self.assertTrue(dataset.cache_composites)  # small subset, replayed every epoch

    def test_cache_is_off_for_full_training_and_exact_when_on(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._build_tree(Path(directory))
            self.assertFalse(self._dataset(root).cache_composites)
            cached = self._dataset(root, overfit_samples=4, cache_composites=True)
            first = cached[0][0].clone()
            self.assertTrue(torch.equal(cached[0][0], first))

    def test_missing_pieces_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._build_tree(Path(directory))
            self.assertEqual(len(self._dataset(root, split="test")), 4)
            with self.assertRaises(ValueError):
                self._dataset(root, split="validation")
            with self.assertRaises(FileNotFoundError):
                self._dataset(root, background_dir=str(Path(directory) / "nope"))
            (root / "Train" / "GT" / "fg000.png").unlink()
            with self.assertRaises(FileNotFoundError):
                self._dataset(root)
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError):
                self._dataset(Path(empty))

    def test_shared_subset_helper_matches_the_am2k_method(self):
        # AM2KMattingDataset keeps its own copy because trained adapters pin
        # their subset by sample id; this guards the two from drifting.
        records = [{"sample_id": f"s{i}", "category": f"c{i % 20}"} for i in range(1800)]
        for seed in (2025, 7, 99):
            for count in (16, 32, 64):
                self.assertEqual(
                    AM2KMattingDataset._category_stratified_subset(records, count, seed),
                    _group_stratified_subset(records, count, seed, key="category"),
                )


class ReportLayoutTests(unittest.TestCase):
    def test_layouts_and_sources_cover_both_datasets(self):
        from matting_report import DATASET_LAYOUTS, build_pair_source

        self.assertEqual(DATASET_LAYOUTS["am2k"]["image"], ("original", ".jpg"))
        self.assertTrue(DATASET_LAYOUTS["d646"]["composite"])
        self.assertEqual(DATASET_LAYOUTS["d646"]["splits"]["train"], "Train")

        with tempfile.TemporaryDirectory() as directory:
            root = Distinctions646Tests._build_tree(Path(directory))
            source = build_pair_source(DATASET_LAYOUTS["d646"], root, "train", 8, root / "backgrounds")
            ids = source.sample_ids()
            self.assertEqual(len(ids), 32)
            image, alpha = source.load(ids[0], 8)
            self.assertEqual(image.size, (8, 8))
            self.assertEqual(alpha.shape, (8, 8))
            self.assertTrue(0.0 <= float(alpha.min()) and float(alpha.max()) <= 1.0)

            paired = build_pair_source(DATASET_LAYOUTS["am2k"], root, "train", 8)
            self.assertFalse(hasattr(paired, "dataset"))


class SubsetManifestTests(unittest.TestCase):
    """The manifest must not assume one dataset's record keys -- assuming
    AM-2K's `image_path` crashed the first D-646 run at startup."""

    class _Fake:
        def __init__(self, records):
            self.dataset = records

    def test_paired_dataset_records_image_paths(self):
        dataset = self._Fake([
            {"sample_id": "a", "category": "cat", "image_path": "/x/a.jpg", "alpha_path": "/x/a.png"}
        ])
        records = _subset_records(dataset)
        self.assertEqual(records[0]["image"], "/x/a.jpg")
        self.assertEqual(records[0]["alpha"], "/x/a.png")
        manifest = _subset_manifest(records, "AM2KMattingDataset")
        self.assertEqual(manifest["images"], ["/x/a.jpg"])

    def test_composited_dataset_has_no_image_paths(self):
        dataset = self._Fake([
            {
                "sample_id": "fg_0",
                "category": "fg",
                "foreground_path": "/x/FG/fg.png",
                "alpha_path": "/x/GT/fg.png",
                "background": "bg000.jpg",
            }
        ])
        records = _subset_records(dataset)
        self.assertEqual(records[0]["foreground"], "/x/FG/fg.png")
        self.assertEqual(records[0]["background"], "bg000.jpg")
        self.assertNotIn("image", records[0])
        manifest = _subset_manifest(records, "Distinctions646MattingDataset")
        # No composite exists on disk, so nothing may be advertised as an image.
        self.assertEqual(manifest["images"], [])
        self.assertEqual(manifest["dataset"], "Distinctions646MattingDataset")

    def test_manifest_serialises_for_the_real_datasets(self):
        import json

        for name, dataset in (
            ("AM2KMattingDataset", AM2KMattingDataset(
                data_dir=["/scratch/mridul/data/matting/am-2k"], resolution=64,
                extra={"split": "train", "overfit_samples": 4, "overfit_seed": 2025})),
            ("Distinctions646MattingDataset", Distinctions646MattingDataset(
                data_dir=["/scratch/mridul/data/matting/distinctions-646"], resolution=64,
                extra={"split": "train", "overfit_samples": 4, "overfit_seed": 2025})),
        ):
            with self.subTest(dataset=name):
                manifest = _subset_manifest(_subset_records(dataset), name)
                self.assertEqual(len(manifest["records"]), 4)
                json.dumps(manifest)


class PreviewRotationTests(unittest.TestCase):
    def test_rotation_varies_by_step_without_repeating_within_a_grid(self):
        grids = [_rotating_preview_indices(32, 4, step) for step in (50, 100, 150, 200)]
        for grid in grids:
            self.assertEqual(len(set(grid)), 4)
            self.assertTrue(all(0 <= index < 32 for index in grid))
        self.assertEqual(len({tuple(grid) for grid in grids}), 4)

    def test_rotation_is_reproducible_and_clamped(self):
        self.assertEqual(_rotating_preview_indices(32, 4, 50), _rotating_preview_indices(32, 4, 50))
        # Asking for more examples than exist must not raise.
        self.assertEqual(len(_rotating_preview_indices(3, 8, 1)), 3)


if __name__ == "__main__":
    unittest.main()

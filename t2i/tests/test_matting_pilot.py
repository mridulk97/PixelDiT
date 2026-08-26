from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
T2I_ROOT = REPO_ROOT / "t2i"
for directory in (str(REPO_ROOT), str(T2I_ROOT)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from diffusion.data.datasets.pixdit_datasets import AM2KMattingDataset
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


if __name__ == "__main__":
    unittest.main()

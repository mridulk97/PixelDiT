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
from train_matting import _sample_training_grid, _validation_losses


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
    def _trainer(mode):
        config = SimpleNamespace(
            model=SimpleNamespace(
                conditioning_mode=mode,
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

    def test_channel_concat_uses_symmetric_sqrt2_expansion(self):
        torch.manual_seed(11)
        base = self._trainer("none")
        state = base.state_dict()
        widened = self._trainer("both")
        result = widened.load_state_dict(state, strict=False)
        self.assertEqual(result.unexpected_keys, [])
        for key in ("core.s_embedder.proj.weight", "core.pixel_embedder.proj.weight"):
            old = state[key]
            new = widened.state_dict()[key]
            self.assertTrue(torch.equal(new[:, : old.shape[1]], old / np.sqrt(2.0)))
            self.assertTrue(torch.equal(new[:, old.shape[1] :], old / np.sqrt(2.0)))
        self.assertTrue(
            torch.equal(widened.core.s_embedder.proj.bias, state["core.s_embedder.proj.bias"])
        )

    def test_sequence_projection_is_unchanged_and_shared(self):
        torch.manual_seed(13)
        base = self._trainer("none")
        state = base.state_dict()
        sequence = self._trainer("sequence")
        result = sequence.load_state_dict(state, strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(result.missing_keys, ["core.reference_type_embedding"])
        self.assertTrue(torch.equal(sequence.core.s_embedder.proj.weight, state["core.s_embedder.proj.weight"]))
        self.assertEqual(sequence.core.s_embedder.proj.in_features, 12)
        self.assertTrue(torch.count_nonzero(sequence.core.reference_type_embedding) == 0)


class LoRATests(unittest.TestCase):
    def test_generated_triplet_grid(self):
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
        correct, shuffled = _validation_losses(
            model,
            flow_matching,
            batch,
            torch.randn(1, 5, 16),
            torch.ones(1, 5),
            1000,
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
        for mode in ("patch", "pixel", "both", "sequence"):
            with self.subTest(mode=mode):
                model = TinyWrapper(mode)
                configure_matting_trainable_parameters(model, rank=2, alpha=2, dropout=0)
                model.core.grad_checkpointing = mode == "sequence"
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

    def test_matting_metrics(self):
        target = np.zeros((12, 12), dtype=np.float32)
        target[3:9, 3:9] = 1.0
        perfect = compute_matting_metrics(target, target)
        self.assertTrue(all(value == 0.0 for value in perfect.values()))
        wrong = compute_matting_metrics(np.zeros_like(target), target)
        self.assertGreater(wrong["sad"], 0.0)
        self.assertGreater(wrong["mse"], 0.0)
        self.assertGreater(wrong["mad"], 0.0)
        self.assertGreater(wrong["gradient"], 0.0)
        self.assertGreater(wrong["connectivity"], 0.0)


if __name__ == "__main__":
    unittest.main()

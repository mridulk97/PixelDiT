import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from typing import Tuple

from .pixeldit_c2i import PatchTokenEmbedder, PixelTokenEmbedder, PiTBlock
from .modules import (
    FinalLayer,
    FeedForward,
    RMSNorm,
    TimestepConditioner,
    apply_adaln,
    apply_rotary_emb,
    precompute_freqs_cis_2d,
    scaled_dot_product_attention_compat,
)


class MMDiTJointAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm_x = RMSNorm(self.head_dim)
        self.k_norm_x = RMSNorm(self.head_dim)
        self.q_norm_y = RMSNorm(self.head_dim)
        self.k_norm_y = RMSNorm(self.head_dim)

        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop_x = nn.Dropout(proj_drop)
        self.proj_drop_y = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            pos_img: torch.Tensor,
            pos_txt: torch.Tensor = None,
            attn_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Nx, C = x.shape
        By, Ny, Cy = y.shape
        assert B == By and C == Cy, "x and y must share batch and channel dims"

        qkv_x = self.qkv_x(x).reshape(B, Nx, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qx, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]
        qx = self.q_norm_x(qx)
        kx = self.k_norm_x(kx)

        qkv_y = self.qkv_y(y).reshape(B, Ny, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qy, ky, vy = qkv_y[0], qkv_y[1], qkv_y[2]
        qy = self.q_norm_y(qy)
        ky = self.k_norm_y(ky)

        qx, kx = apply_rotary_emb(qx, kx, freqs_cis=pos_img)
        if pos_txt is not None:
            qy, ky = apply_rotary_emb(qy, ky, freqs_cis=pos_txt)

        qx = qx.transpose(1, 2)
        kx = kx.transpose(1, 2)
        vx = vx.transpose(1, 2)

        qy = qy.transpose(1, 2)
        ky = ky.transpose(1, 2)
        vy = vy.transpose(1, 2)

        q_joint = torch.cat([qy, qx], dim=2)
        k_joint = torch.cat([ky, kx], dim=2)
        v_joint = torch.cat([vy, vx], dim=2)

        out_joint = scaled_dot_product_attention_compat(
            q_joint,
            k_joint,
            v_joint,
            dropout_p=0.0,
            attn_mask=attn_mask,
        )
        out_y = out_joint[:, :, :Ny, :]
        out_x = out_joint[:, :, Ny:, :]

        out_y = out_y.transpose(1, 2).reshape(B, Ny, C)
        out_x = out_x.transpose(1, 2).reshape(B, Nx, C)

        out_x = self.proj_drop_x(self.proj_x(out_x))
        out_y = self.proj_drop_y(self.proj_y(out_y))
        return out_x, out_y


class MMDiTBlockT2I(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4.0, adaLN_modulation_img=None, adaLN_modulation_txt=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.groups = groups
        self.head_dim = hidden_size // groups

        self.norm_x1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y1 = RMSNorm(hidden_size, eps=1e-6)

        self.attn = MMDiTJointAttention(hidden_size, num_heads=groups, qkv_bias=False)

        self.norm_x2 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y2 = RMSNorm(hidden_size, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_x = FeedForward(hidden_size, mlp_hidden_dim)
        self.mlp_y = FeedForward(hidden_size, mlp_hidden_dim)

        self.adaLN_modulation_img = adaLN_modulation_img if adaLN_modulation_img is not None else nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_modulation_txt = adaLN_modulation_txt if adaLN_modulation_txt is not None else nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, y, c, pos_img, pos_txt=None, attn_mask=None):
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_img(c).chunk(6, dim=-1)
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = self.adaLN_modulation_txt(c).chunk(6, dim=-1)

        x_norm = apply_adaln(self.norm_x1(x), shift_msa_x, scale_msa_x)
        y_norm = apply_adaln(self.norm_y1(y), shift_msa_y, scale_msa_y)
        attn_x, attn_y = self.attn(x_norm, y_norm, pos_img, pos_txt, attn_mask)
        x = x + gate_msa_x * attn_x
        y = y + gate_msa_y * attn_y

        x = x + gate_mlp_x * self.mlp_x(apply_adaln(self.norm_x2(x), shift_mlp_x, scale_mlp_x))
        y = y + gate_mlp_y * self.mlp_y(apply_adaln(self.norm_y2(y), shift_mlp_y, scale_mlp_y))
        return x, y


class RefinementHead(nn.Module):
    """Pixel-resolution refinement applied after the fold.

    Every learned layer in PixDiT couples pixels either one at a time (the
    per-pixel MLP, and a ``final_layer`` of 67 parameters) or a whole
    ``patch_size**2`` block at a time (``compress_to_attn`` collapses the patch
    to a single token, attention runs at patch resolution, ``expand_from_attn``
    writes it back). Two pixels that touch across a patch boundary therefore
    have no path to each other except a round trip through both patch tokens.
    Folding the residual error onto ``(y % 16, x % 16)`` shows 13.7% variation
    across in-patch positions against 2.7% at a control period of 15, so the
    error really is locked to the patch grid.

    These convolutions run on the folded image, which is the first point in the
    network where neighbouring pixels are actually adjacent. They see the
    pixel-branch features rather than only the 3-channel projection of them,
    the coarse output, and -- for matting, the input that matters most -- the
    full-resolution conditioning image, which already contains the hair and
    whisker detail the patch stream cannot carry.

    The last convolution is zero-initialized, so the head starts as the
    identity and the wrapped model reproduces its pretrained output exactly at
    step 0, the same discipline as a zero-initialized LoRA branch.
    """

    def __init__(
        self,
        pixel_hidden_size: int,
        out_channels: int,
        guide_channels: int = 0,
        width: int = 64,
        dilations: Tuple[int, ...] = (1, 2, 4, 1),
    ):
        super().__init__()
        dilations = tuple(int(dilation) for dilation in dilations)
        if not dilations:
            raise ValueError("RefinementHead needs at least one dilated convolution")
        if any(dilation < 1 for dilation in dilations):
            raise ValueError(f"RefinementHead dilations must be >= 1, got {dilations}")
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.out_channels = int(out_channels)
        self.guide_channels = int(guide_channels)
        self.width = int(width)
        self.dilations = dilations
        self.in_channels = self.pixel_hidden_size + self.out_channels + self.guide_channels

        layers = []
        channels = self.in_channels
        for dilation in dilations:
            layers.append(
                nn.Conv2d(channels, self.width, 3, padding=dilation, dilation=dilation)
            )
            layers.append(nn.SiLU())
            channels = self.width
        projection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        nn.init.zeros_(projection.weight)
        nn.init.zeros_(projection.bias)
        layers.append(projection)
        self.body = nn.Sequential(*layers)

    @property
    def receptive_field(self) -> int:
        """Width of the square the head can see, in pixels.

        Worth keeping above ``patch_size`` so every pixel reaches across at
        least one patch seam in each direction.
        """
        return 1 + 2 * (sum(self.dilations) + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class PixDiT_T2I(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_groups=16,
        hidden_size=1152,
        pixel_hidden_size=64,
        pixel_attn_hidden_size=None,
        pixel_num_groups=None,
        patch_depth=26,
        pixel_depth=2,
        num_text_blocks=4,
        patch_size=16,
        txt_embed_dim=4096,
        txt_max_length=1024,
        use_text_rope: bool = True,
        text_rope_theta: float = 10000.0,
        repa_encoder_index: int = -1,
        use_pixel_abs_pos: bool = True,
        pit_adaln_post_modulation: bool = False,
        conditioning_mode: str = "none",
        sequence_rope_mode: str = "aligned",
        sequence_rope_offset: float = None,
        use_sequence_type_embedding: bool = True,
        use_refine_head: bool = False,
        refine_head_width: int = 64,
        refine_head_dilations: Tuple[int, ...] = (1, 2, 4, 1),
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_groups = int(num_groups)
        self.patch_depth = int(patch_depth)
        self.pixel_depth = int(pixel_depth)
        self.num_text_blocks = int(num_text_blocks)
        self.patch_size = int(patch_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.txt_embed_dim = int(txt_embed_dim)
        self.txt_max_length = int(txt_max_length)
        self.use_text_rope = bool(use_text_rope)
        self.text_rope_theta = float(text_rope_theta)
        self.repa_encoder_index = int(repa_encoder_index)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.pit_adaln_post_modulation = bool(pit_adaln_post_modulation)
        self.conditioning_mode = str(conditioning_mode).lower()
        self.sequence_rope_mode = str(sequence_rope_mode).lower()
        self.sequence_rope_offset = (
            None if sequence_rope_offset is None else float(sequence_rope_offset)
        )
        self.use_sequence_type_embedding = bool(use_sequence_type_embedding)
        valid_conditioning_modes = {
            "none",
            "patch",
            "pixel",
            "both",
            "sequence",
            "sequence_pixel",
        }
        if self.conditioning_mode not in valid_conditioning_modes:
            raise ValueError(
                f"conditioning_mode must be one of {sorted(valid_conditioning_modes)}, "
                f"got {self.conditioning_mode!r}"
            )
        if self.sequence_rope_mode not in {"aligned", "offset"}:
            raise ValueError("sequence_rope_mode must be 'aligned' or 'offset'")
        if self.pixel_depth <= 0:
            raise ValueError("PixDiT_T2I expects pixel_depth > 0 to retain the pixel pathway")

        # The reference image can enter through three independent doors, and a
        # mode is just a choice of which doors are open.
        # patch:    widen the patch projection with reference patch vectors.
        # sequence: append reference patch tokens as a second block.
        # pixel:    concatenate reference channels into the pixel branch.
        self.patch_conditioning = self.conditioning_mode in {"patch", "both"}
        self.sequence_conditioning = self.conditioning_mode in {"sequence", "sequence_pixel"}
        self.pixel_conditioning = self.conditioning_mode in {"pixel", "both", "sequence_pixel"}

        patch_input_channels = in_channels * patch_size ** 2
        pixel_input_channels = in_channels
        if self.patch_conditioning:
            patch_input_channels *= 2
        if self.pixel_conditioning:
            pixel_input_channels *= 2

        self.pixel_embedder = PixelTokenEmbedder(
            pixel_input_channels,
            self.pixel_hidden_size,
            use_pixel_abs_pos=self.use_pixel_abs_pos,
        )
        self.s_embedder = PatchTokenEmbedder(patch_input_channels, hidden_size, bias=True)
        if self.sequence_conditioning and self.use_sequence_type_embedding:
            # Under aligned RoPE the target and reference blocks occupy the same
            # positions and share one projection, so these embeddings carry the
            # only signal that tells the two streams apart. initialize_weights
            # gives them a non-zero start; leaving them at zero makes the split
            # degenerate at step 0 and the model has to route on content alone.
            self.target_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.reference_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_size))
        else:
            self.register_parameter("target_type_embedding", None)
            self.register_parameter("reference_type_embedding", None)
        self.t_embedder = TimestepConditioner(hidden_size)
        self.y_embedder = PatchTokenEmbedder(self.txt_embed_dim, hidden_size, bias=True, norm_layer=RMSNorm)
        self.y_pos_embedding = nn.Parameter(torch.randn(1, self.txt_max_length, hidden_size))

        self._shared_cond_adaln = None
        self._shared_cond_adaln_img = None
        self._shared_cond_adaln_txt = None
        self.patch_blocks = nn.ModuleList([
            MMDiTBlockT2I(
                self.hidden_size,
                self.num_groups,
                adaLN_modulation_img=self._shared_cond_adaln_img,
                adaLN_modulation_txt=self._shared_cond_adaln_txt,
            )
            for _ in range(self.patch_depth)
        ])
        self.text_refine_blocks = None
        self.pixel_attn_hidden_size = (
            int(pixel_attn_hidden_size) if pixel_attn_hidden_size is not None else self.hidden_size
        )
        self.pixel_num_groups = int(pixel_num_groups) if pixel_num_groups is not None else self.num_groups
        self.pixel_blocks = nn.ModuleList(
            [
                PiTBlock(
                    self.pixel_hidden_size,
                    self.hidden_size,
                    patch_size=self.patch_size,
                    num_heads=self.num_groups,
                    mlp_ratio=4.0,
                    attn_hidden_size=self.pixel_attn_hidden_size,
                    attn_num_heads=self.pixel_num_groups,
                    rope_fn=precompute_freqs_cis_2d,
                    adaln_post_modulation=self.pit_adaln_post_modulation,
                )
                for _ in range(self.pixel_depth)
            ]
        )

        self.final_layer = FinalLayer(self.pixel_hidden_size, self.out_channels)

        # The head is the only module that couples neighbouring pixels across a
        # patch seam. It is fed the conditioning image when one exists, which is
        # what makes it a guided filter rather than a blur.
        if use_refine_head:
            self.refine_head = RefinementHead(
                self.pixel_hidden_size,
                self.out_channels,
                guide_channels=self.in_channels if self.conditioning_mode != "none" else 0,
                width=refine_head_width,
                dilations=refine_head_dilations,
            )
        else:
            self.refine_head = None

        self.precompute_pos = dict()
        self.precompute_pos_txt = dict()
        self.last_repa_tokens = None

        self.initialize_weights()

    def fetch_pos(self, height, width, device, offset=(0.0, 0.0)):
        key = (height, width, float(offset[0]), float(offset[1]))
        if key in self.precompute_pos:
            return self.precompute_pos[key].to(device)
        else:
            pos = precompute_freqs_cis_2d(
                self.hidden_size // self.num_groups,
                height,
                width,
                offset=offset,
            ).to(device)
            self.precompute_pos[key] = pos
            return pos

    def _sequence_positions(self, height, width, device):
        target_pos = self.fetch_pos(height, width, device)
        if self.sequence_rope_mode == "aligned":
            reference_pos = target_pos
        else:
            if self.sequence_rope_offset is None:
                # PixelDiT uses [0, 16] (inclusive). Move the second grid by
                # one complete grid plus one grid step so no columns overlap.
                step = 16.0 / max(width - 1, 1)
                x_offset = 16.0 + step
            else:
                x_offset = self.sequence_rope_offset
            reference_pos = self.fetch_pos(height, width, device, offset=(0.0, x_offset))
        return torch.cat([target_pos, reference_pos], dim=0)

    def _fold_pixel_tokens(self, tokens, batch, length, height, width, channels):
        """Fold ``[B*L, patch_size**2, channels]`` back into ``[B, channels, H, W]``."""
        patch_area = self.patch_size * self.patch_size
        tokens = tokens.view(batch, length, patch_area, channels)
        tokens = tokens.permute(0, 3, 2, 1).contiguous().view(batch, channels * patch_area, length)
        return torch.nn.functional.fold(
            tokens,
            (height, width),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def fetch_pos_text(self, length, device):
        if length in self.precompute_pos_txt:
            return self.precompute_pos_txt[length].to(device)
        head_dim = self.hidden_size // self.num_groups
        freqs = 1.0 / (self.text_rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        positions = torch.arange(0, length, device=device).float().unsqueeze(1)
        angles = positions * freqs.unsqueeze(0)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        self.precompute_pos_txt[length] = freqs_cis
        return freqs_cis

    def initialize_weights(self):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        for type_embedding in (self.target_type_embedding, self.reference_type_embedding):
            if type_embedding is not None:
                nn.init.normal_(type_embedding, std=0.02)

    def forward(self, x, t, y, condition_image=None, s=None, mask=None):
        B, _, H, W = x.shape
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(
                f"Image size {(H, W)} must be divisible by patch_size={self.patch_size}"
            )
        Hs = H // self.patch_size
        Ws = W // self.patch_size
        L = Hs * Ws

        if self.conditioning_mode != "none":
            if condition_image is None:
                raise ValueError(
                    f"condition_image is required for conditioning_mode={self.conditioning_mode!r}"
                )
            if condition_image.dim() != 4:
                raise ValueError("condition_image must have shape [B, 3, H, W]")
            if condition_image.shape != x.shape:
                raise ValueError(
                    f"condition_image shape {tuple(condition_image.shape)} must match "
                    f"the noisy target shape {tuple(x.shape)}"
                )
            condition_image = condition_image.to(device=x.device, dtype=x.dtype)

        pos = (
            self._sequence_positions(Hs, Ws, x.device)
            if self.sequence_conditioning
            else self.fetch_pos(Hs, Ws, x.device)
        )
        x_patches = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        condition_patches = None
        if condition_image is not None and (self.patch_conditioning or self.sequence_conditioning):
            condition_patches = torch.nn.functional.unfold(
                condition_image,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ).transpose(1, 2)

        t_emb = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)

        if y.dim() != 3:
            raise ValueError("Text embedding y must be [B, L, D]")
        Ltxt = min(y.shape[1], self.txt_max_length)
        y = y[:, :Ltxt, :]
        y_emb = self.y_embedder(y).view(B, Ltxt, self.hidden_size)
        y_emb = y_emb + self.y_pos_embedding[:, :Ltxt, :].to(y_emb.dtype)

        time_condition = torch.nn.functional.silu(t_emb)

        if s is None:
            if self.patch_conditioning:
                s0 = self.s_embedder(torch.cat([x_patches, condition_patches], dim=-1))
            elif self.sequence_conditioning:
                target_tokens = self.s_embedder(x_patches)
                reference_tokens = self.s_embedder(condition_patches)
                if self.target_type_embedding is not None:
                    target_tokens = target_tokens + self.target_type_embedding.to(
                        dtype=target_tokens.dtype
                    )
                if self.reference_type_embedding is not None:
                    reference_tokens = reference_tokens + self.reference_type_embedding.to(
                        dtype=reference_tokens.dtype
                    )
                s0 = torch.cat([target_tokens, reference_tokens], dim=1)
            else:
                s0 = self.s_embedder(x_patches)
            pos_txt = self.fetch_pos_text(Ltxt, x.device) if self.use_text_rope else None
            attn_mask_joint = None
            if mask is not None and isinstance(mask, torch.Tensor):
                m = mask
                while m.dim() > 2 and m.size(1) == 1:
                    m = m.squeeze(1)
                if m.dim() == 3 and m.size(1) == 1:
                    m = m.squeeze(1)
                if m.dim() == 2:
                    pad = (m == 0)
                    image_length = 2 * L if self.sequence_conditioning else L
                    pad_img = torch.zeros((B, image_length), dtype=torch.bool, device=x.device)
                    attn_mask_joint = torch.cat([pad[:, :Ltxt], pad_img], dim=1).view(
                        B, 1, 1, Ltxt + image_length
                    )
            self.last_repa_tokens = None
            s = s0
            use_checkpoint = self.training and bool(getattr(self, "grad_checkpointing", False))
            for i in range(self.patch_depth):
                if use_checkpoint:
                    s, y_emb = checkpoint(
                        self.patch_blocks[i],
                        s,
                        y_emb,
                        time_condition,
                        pos,
                        pos_txt,
                        attn_mask_joint,
                        use_reentrant=False,
                    )
                else:
                    s, y_emb = self.patch_blocks[i](
                        s,
                        y_emb,
                        time_condition,
                        pos,
                        pos_txt,
                        attn_mask_joint,
                    )
                if 0 < self.repa_encoder_index == (i + 1):
                    self.last_repa_tokens = s[:, :L, :]
            s = torch.nn.functional.silu(t_emb + s)
            if self.sequence_conditioning:
                s = s[:, :L, :]
        if not (0 < self.repa_encoder_index <= self.patch_depth):
            self.last_repa_tokens = s[:, :L, :]

        batch_size, length, _ = s.shape
        if length != L:
            if length > L:
                s = s[:, :L, :]
            else:
                pad_len = L - length
                s = torch.cat([s, s.new_zeros(B, pad_len, s.shape[2])], dim=1)
            length = L

        # Sequence mode slices the target half from a larger token tensor;
        # reshape also handles that intentionally non-contiguous view.
        s_cond = s.reshape(B * L, self.hidden_size)
        pixel_inputs = (
            torch.cat([x, condition_image], dim=1)
            if self.pixel_conditioning
            else x
        )
        x_pixels = self.pixel_embedder(
            pixel_inputs,
            img_height=H,
            img_width=W,
            patch_size=self.patch_size,
        )
        for blk in self.pixel_blocks:
            if self.training and bool(getattr(self, "grad_checkpointing", False)):
                x_pixels = checkpoint(
                    blk,
                    x_pixels,
                    s_cond,
                    H,
                    W,
                    self.patch_size,
                    mask,
                    use_reentrant=False,
                )
            else:
                x_pixels = blk(x_pixels, s_cond, H, W, self.patch_size, mask)

        # Fold the pixel-branch features too, not just the projection of them:
        # final_layer is 67 parameters, so almost everything the pixel blocks
        # computed is discarded before the tensor ever becomes an image.
        features = (
            self._fold_pixel_tokens(x_pixels, B, L, H, W, self.pixel_hidden_size)
            if self.refine_head is not None
            else None
        )

        x_pixels = self.final_layer(x_pixels)
        x_img = self._fold_pixel_tokens(x_pixels, B, L, H, W, self.out_channels)

        if self.refine_head is not None:
            head_inputs = [features, x_img]
            if self.refine_head.guide_channels:
                head_inputs.append(condition_image.to(x_img.dtype))
            # Residual, with the head's last convolution zero-initialized: this
            # is exactly the pretrained output until the head learns otherwise.
            x_img = x_img + self.refine_head(torch.cat(head_inputs, dim=1))
        return x_img

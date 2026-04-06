# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Transformer configuration for native FastVideo integration.
"""

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig

import re


def is_ltx2_blocks(name: str, _module) -> bool:
    res = re.search(r"(?:^|\.)transformer_blocks\.\d+$", name) is not None
    return res


@dataclass
class LTX2VideoArchConfig(DiTArchConfig):
    """Architecture configuration for LTX-2 video transformer."""

    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_ltx2_blocks])
    _compile_conditions: list = field(default_factory=lambda: [is_ltx2_blocks])

    # Parameter name mapping for weight conversion (hf/comfy -> FastVideo)
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^model\.diffusion_model\.(.*)$": r"model.\1",
            r"^diffusion_model\.(.*)$": r"model.\1",
            r"^model\.(.*)$": r"model.\1",
            r"^(.*)$": r"model.\1",
        })

    reverse_param_names_mapping: dict = field(default_factory=lambda: {})
    lora_param_names_mapping: dict = field(default_factory=lambda: {})

    # Core transformer settings (defaults from LTX-2 metadata)
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    num_layers: int = 48
    cross_attention_dim: int = 4096
    caption_channels: int = 3840
    norm_eps: float = 1e-6
    attention_type: str = "default"
    rope_type: str = "split"
    double_precision_rope: bool = True

    positional_embedding_theta: float = 10000.0
    positional_embedding_max_pos: list[int] = field(default_factory=lambda: [20, 2048, 2048])
    timestep_scale_multiplier: int = 1000
    use_middle_indices_grid: bool = True

    # Patchification (video-only path)
    patch_size: tuple[int, int, int] = (1, 1, 1)
    num_channels_latents: int = 128
    in_channels: int | None = None
    out_channels: int | None = None

    # Audio defaults (reserved for joint AV ports)
    audio_num_attention_heads: int = 32
    audio_attention_head_dim: int = 64
    audio_in_channels: int = 128
    audio_out_channels: int = 128
    audio_cross_attention_dim: int = 2048
    audio_positional_embedding_max_pos: list[int] = field(default_factory=lambda: [20])
    av_ca_timestep_scale_multiplier: int = 1

    # LTX-2.3 (22B) specific features
    caption_proj_before_connector: bool = False  # True for 22B, False for 19B
    cross_attention_adaln: bool = False          # True for 22B, False for 19B
    apply_gated_attention: bool = False          # True for 22B, False for 19B
    model_version: str = "19b"                   # "19b" or "22b"

    def __post_init__(self):
        super().__post_init__()
        patch_volume = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        if self.in_channels is None:
            self.in_channels = self.num_channels_latents * patch_volume
        if self.out_channels is None:
            self.out_channels = self.in_channels

        # Auto-configure based on model version
        if self.model_version == "22b":
            self.caption_proj_before_connector = True
            self.cross_attention_adaln = True
            self.apply_gated_attention = True


@dataclass
class LTX2VideoConfig(DiTConfig):
    """Main configuration for LTX-2 transformer."""

    arch_config: DiTArchConfig = field(default_factory=LTX2VideoArchConfig)
    prefix: str = "ltx2"


@dataclass
class LTX23VideoArchConfig(LTX2VideoArchConfig):
    """Architecture configuration for LTX-2.3 (22B) video transformer.

    Key differences from 19B:
    - caption_proj_before_connector=True: projection lives in
      text encoder connector, not in the transformer.
    - cross_attention_adaln=True: text cross-attention uses
      AdaLN modulation (scale_shift_table grows from 6→9).
    - apply_gated_attention=True: optional gate projection on
      attention output.
    """

    model_version: str = "22b"
    caption_proj_before_connector: bool = True
    cross_attention_adaln: bool = True
    apply_gated_attention: bool = True


@dataclass
class LTX23VideoConfig(DiTConfig):
    """Main configuration for LTX-2.3 (22B) transformer."""

    arch_config: DiTArchConfig = field(
        default_factory=LTX23VideoArchConfig)
    prefix: str = "ltx23"

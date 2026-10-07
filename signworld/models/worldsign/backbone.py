"""The video encoder: V-JEPA 2.1 ViT-L, frozen, with LoRA on every block (§4.4.1, §4.4.2).

It serves both passes. The physical pass encodes only the visible tokens of a mask and needs
the raw outputs of the hierarchical blocks (6/12/18/24), read with forward hooks so Meta's
forward runs unchanged. The semantic pass encodes the whole clip and needs the last block's
normalised output, which is what the encoder returns. An encoder that no level trains (ESP-2,
where the semantic level reads it without gradient) gets no LoRA and keeps its norms frozen.

Meta's attention runs without cuDNN's kernel (``drop_cudnn_attention``): with torch 2.14 and
cuDNN 9.24 on H100 its backward returns NaN on the encoder's visible tokens (P13).
"""

from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Any

import torch
from torch import Tensor, nn

from signworld.experiment.train.config import EncoderSettings
from signworld.models.encoders.video_encoders import model_input

from . import lora


class VideoBackbone(nn.Module):
    def __init__(
        self, encoder: nn.Module, settings: EncoderSettings, *, adapted: bool = True
    ) -> None:
        super().__init__()
        drop_cudnn_attention()
        # Meta's module is untyped and carries plain attributes (blocks, norms_block, ...).
        released: Any = encoder.requires_grad_(False)
        released.use_activation_checkpointing = settings.activation_checkpointing
        self.encoder: Any = released
        self.settings = settings
        levels = list(settings.levels)
        hierarchical = list(getattr(encoder, "hierarchical_layers", levels))
        if levels != hierarchical:
            raise ValueError(f"levels {levels} differ from the encoder's own {hierarchical}")
        self.levels = levels
        self.adapted = adapted
        self.adapters: list[Any] = []
        if not adapted:
            return
        self.adapters = lora.inject(
            self._blocks(), settings.lora.targets, settings.lora.rank, settings.lora.alpha
        )
        if settings.train_norms:
            for module in self._blocks().modules():
                if isinstance(module, nn.LayerNorm):
                    module.requires_grad_(True)

    def _blocks(self) -> nn.ModuleList:
        blocks: nn.ModuleList = self.encoder.blocks
        return blocks

    @property
    def width(self) -> int:
        return int(self.encoder.embed_dim)

    def level_norms(self) -> Sequence[nn.LayerNorm]:
        """The encoder's per-level LayerNorms, which the fusion copies (PC6)."""
        return list(self.encoder.norms_block)

    def context_levels(self, frames: Tensor, context: Tensor) -> list[Tensor]:
        """Raw outputs of the hierarchical blocks for the visible tokens only.

        ``frames`` is (batch, T, H, W, 3) uint8; ``context`` (batch, n) token indices. Masked
        tokens are removed from the encoder's input, not zeroed (§4.6).
        """
        captured: dict[int, Tensor] = {}
        blocks = self._blocks()
        handles = [
            blocks[index].register_forward_hook(_capture(captured, index)) for index in self.levels
        ]
        try:
            self.encoder(model_input(frames), masks=[context])
        finally:
            for handle in handles:
                handle.remove()
        return [captured[index] for index in self.levels]

    def tokens(self, frames: Tensor) -> Tensor:
        """(batch, N, width) normalised output of the last block on the whole clip."""
        output: Tensor = self.encoder(model_input(frames))
        return output


def drop_cudnn_attention() -> None:
    """Keep cuDNN out of the attention of Meta's encoder and predictor.

    Meta's blocks call ``scaled_dot_product_attention`` inside ``torch.backends.cuda.sdp_kernel()``,
    which enables every backend, cuDNN included, so ``enable_cudnn_sdp(False)`` would not reach
    them. On H100 with torch 2.14 and cuDNN 9.24 torch picks cuDNN there, and its backward
    returns NaN on the encoder's visible tokens from the first step (P13, 7/10). The helper is
    replaced by the same one with cuDNN off: flash attention serves instead. Our own predictors
    do not call it and keep torch's choice. Idempotent.
    """
    current = torch.backends.cuda.sdp_kernel
    if getattr(current, "drops_cudnn", False):
        return

    def sdp_kernel(
        enable_flash: bool = True,
        enable_math: bool = True,
        enable_mem_efficient: bool = True,
        enable_cudnn: bool = True,  # ignored: cuDNN stays off
    ) -> AbstractContextManager[Any]:
        return current(
            enable_flash=enable_flash,
            enable_math=enable_math,
            enable_mem_efficient=enable_mem_efficient,
            enable_cudnn=False,
        )

    sdp_kernel.drops_cudnn = True  # type: ignore[attr-defined]
    torch.backends.cuda.sdp_kernel = sdp_kernel  # type: ignore[assignment]


def _capture(store: dict[int, Tensor], index: int):  # type: ignore[no-untyped-def]
    def hook(_module: nn.Module, _inputs: object, output: object) -> None:
        store[index] = output[0] if isinstance(output, tuple | list) else output  # type: ignore[assignment]

    return hook

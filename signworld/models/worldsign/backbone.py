"""The video encoder: V-JEPA 2.1 ViT-L, frozen, with LoRA on every block (§4.4.1, §4.4.2).

It serves both passes. The physical pass encodes only the visible tokens of a mask and needs
the raw outputs of the hierarchical blocks (6/12/18/24), read with forward hooks so Meta's
forward runs unchanged. The semantic pass encodes the whole clip and needs the last block's
normalised output, which is what the encoder returns. An encoder that no level trains (ESP-2,
where the semantic level reads it without gradient) gets no LoRA and keeps its norms frozen.

Meta's attention runs without cuDNN's kernel (``drop_cudnn_attention``): with torch 2.14 and
cuDNN 9.24 on H100 its backward returns NaN on the encoder's visible tokens (P13).
``compile_blocks`` compiles each block's ``forward`` (``EncoderSettings.compile_blocks``).
"""

import types
from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from signworld.experiment.train.config import EncoderSettings
from signworld.models.encoders.video_encoders import model_input

from . import lora, meta_rope


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
        if adapted:
            self.adapters = lora.inject(
                self._blocks(), settings.lora.targets, settings.lora.rank, settings.lora.alpha
            )
            if settings.train_norms:
                for module in self._blocks().modules():
                    if isinstance(module, nn.LayerNorm):
                        module.requires_grad_(True)
        if settings.compile_blocks:  # after the LoRA, which replaces the blocks' layers
            compile_blocks(self._blocks())

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
    replaced by ``torch.nn.attention.sdpa_kernel`` over the same backends without cuDNN: flash
    attention serves instead. Unlike the deprecated helper, ``sdpa_kernel`` is traced by
    ``torch.compile`` without a graph break (one graph per block instead of four). Our own
    predictors do not call it and keep torch's choice. Idempotent.
    """
    if getattr(torch.backends.cuda.sdp_kernel, "drops_cudnn", False):
        return

    def sdp_kernel(
        enable_flash: bool = True,
        enable_math: bool = True,
        enable_mem_efficient: bool = True,
        enable_cudnn: bool = True,  # ignored: cuDNN stays off
    ) -> AbstractContextManager[Any]:
        chosen = [
            backend
            for backend, enabled in (
                (SDPBackend.FLASH_ATTENTION, enable_flash),
                (SDPBackend.EFFICIENT_ATTENTION, enable_mem_efficient),
                (SDPBackend.MATH, enable_math),
            )
            if enabled
        ]
        return sdpa_kernel(chosen)

    sdp_kernel.drops_cudnn = True  # type: ignore[attr-defined]
    torch.backends.cuda.sdp_kernel = sdp_kernel  # type: ignore[assignment]


RECOMPILE_LIMIT = 64
"""Graphs per compiled ``forward``: every shape (whole clip, each mask kind's visible tokens,
the last validation batch), with and without gradient, encoder and predictor blocks."""


def compile_blocks(blocks: nn.Module) -> None:
    """``torch.compile`` on each block's ``forward``, not on ``__call__``.

    The forward hooks of ``context_levels`` and the activation checkpointing of Meta's loop
    stay outside the compiled code, unchanged. One compiled function per block class serves
    every block of it (their weights are inputs of the graph, not constants). The compiled
    function is bound to the block as a method, so a ``deepcopy`` rebinds it to the copy.

    Three conditions keep the checkpointing's recomputation on the graph of the forward (else
    ``CheckpointError``, pytorch#166926):

    * the batch and the token count are dynamic from the first call (``_dynamic``): the
      visible tokens of a mask change at every step, and a static graph that turns dynamic
      leaves two graphs matching the same input;
    * every call of this function compiles its own copy of the code (``_own_code``): dynamo
      keeps graphs per code object, and the encoder's and the predictor's blocks are both
      Meta's ``Block``;
    * the recomputation runs in autograd's thread, whose intra-op thread count differs from
      the main thread's: dynamo guards on it and would compile a new graph inside the
      backward, so each call first sets the count of the thread that compiled.

    Inductor draws its random numbers like eager (``fallback_random``): the recomputation must
    replay the predictor's dropout masks of the forward exactly.
    """
    drop_cudnn_attention()
    if meta_rope.ENABLED:  # one rotation per q or k instead of six slices
        meta_rope.fuse_rope(blocks)
    torch._dynamo.config.recompile_limit = max(
        torch._dynamo.config.recompile_limit, RECOMPILE_LIMIT
    )
    torch._inductor.config.fallback_random = True
    compiled: dict[type, Any] = {}
    for block in blocks.children():
        kind = type(block)
        if kind not in compiled:
            compiled[kind] = _dynamic(torch.compile(_own_code(kind.forward)))
        block.forward = types.MethodType(compiled[kind], block)


def _own_code(function: Any) -> types.FunctionType:
    """``function`` with a code object of its own, hence a dynamo cache of its own."""
    copy = types.FunctionType(
        function.__code__.replace(),
        function.__globals__,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    copy.__kwdefaults__ = function.__kwdefaults__
    return copy


def _dynamic(compiled: Any) -> Any:
    """``compiled`` with the batch and token dimensions of ``x`` and ``mask`` marked dynamic,
    run with the intra-op thread count of the thread that builds it."""
    threads = torch.get_num_threads()

    def forward(self: nn.Module, x: Tensor, mask: Any = None, *args: Any, **kwargs: Any) -> Any:
        if torch.get_num_threads() != threads:  # autograd's thread, in the recomputation
            torch.set_num_threads(threads)
        for tensor in (x, mask):
            if isinstance(tensor, Tensor):
                torch._dynamo.maybe_mark_dynamic(tensor, 0)
                torch._dynamo.maybe_mark_dynamic(tensor, 1)
        return compiled(self, x, mask, *args, **kwargs)

    return forward


def _capture(store: dict[int, Tensor], index: int):  # type: ignore[no-untyped-def]
    def hook(_module: nn.Module, _inputs: object, output: object) -> None:
        store[index] = output[0] if isinstance(output, tuple | list) else output  # type: ignore[assignment]

    return hook

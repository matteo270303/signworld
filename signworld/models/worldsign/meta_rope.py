"""Meta's RoPE attention with one rotation of the whole head instead of six slices.

``RoPEAttention.forward`` of V-JEPA 2.1 rotates q and k in three slices (time, row, column),
each one a slice, a ``cat`` and its own sin/cos; the last ``head_dim - 3·slice`` channels pass
unrotated. ``fuse_rope`` replaces the forward of every such module by ``forward``: the same
sin/cos (the same operations, in the same dtype), laid side by side with cos 1 and sin 0 on
the channels that pass, so one product rotates the whole head. Elementwise the arithmetic is
Meta's, ``x·cos + y·sin``, on every channel (``x·1 + y·0 = x``). Under ``torch.compile`` the
rotation becomes one kernel per q or k reading ``qkv`` once, where Meta's took six. Meta's
own forward still serves what this one does not cover (registers, a cls token, the attention
map, no SDPA). Meta's files stay as released.
"""

import types
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

ENABLED = True
"""``compile_blocks`` fuses the rotation of the blocks it compiles (``scripts/perf``)."""


def fuse_rope(module: nn.Module) -> int:
    """Rebind ``forward`` of every RoPE attention under ``module``; returns how many."""
    count = 0
    for child in module.modules():
        if type(child).__name__ == "RoPEAttention" and not getattr(child, "fused_rope", False):
            # Meta's module is untyped: its plain attributes are set as such.
            attention: Any = child
            attention.meta_forward = attention.forward
            attention.forward = types.MethodType(forward, attention)
            attention.fused_rope = True
            count += 1
    return count


def _axis(pos: Tensor, dim: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Meta's ``rotate_queries_or_keys`` sin/cos of one axis, operation for operation."""
    omega = torch.arange(dim // 2, dtype=dtype, device=pos.device)
    omega /= dim / 2.0
    omega = 1.0 / 10000**omega
    freq = torch.einsum("..., f -> ... f", pos, omega)
    return freq.sin().repeat_interleave(2, dim=-1), freq.cos().repeat_interleave(2, dim=-1)


def tables(self: Any, positions: list[Tensor], dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """(..., head_dim) sin and cos of the time, row and column positions, then 0 and 1."""
    sins, coss = [], []
    for pos in positions:
        sin, cos = _axis(pos, self.d_dim, dtype)
        sins.append(sin)
        coss.append(cos)
    rest = self.head_dim - 3 * self.d_dim
    if rest:
        shape = (*sins[0].shape[:-1], rest)
        sins.append(sins[0].new_zeros(shape))
        coss.append(coss[0].new_ones(shape))
    return torch.cat(sins, dim=-1), torch.cat(coss, dim=-1)


def _pairs(x: Tensor) -> Tensor:
    """(-x₂, x₁) of every pair of channels, as Meta's ``y``."""
    y1, y2 = x.reshape(*x.shape[:-1], -1, 2).unbind(dim=-1)
    return torch.stack((-y2, y1), dim=-1).flatten(-2)


def forward(  # noqa: PLR0913, PLR0917 (Meta's signature)
    self: Any,
    x: Tensor,
    mask: Tensor | None = None,
    T: int | None = None,  # noqa: N803 (Meta's name)
    H_patches: int | None = None,  # noqa: N803
    W_patches: int | None = None,  # noqa: N803
    return_attn: bool = False,
) -> Any:
    uniform = self.d_dim == self.h_dim == self.w_dim
    if return_attn or not self.use_sdpa or self.n_registers or self.has_cls_first or not uniform:
        return self.meta_forward(x, mask, T, H_patches, W_patches, return_attn)
    B, N, C = x.size()  # noqa: N806 (Meta's names)
    qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if mask is not None:
        mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
    else:
        grid_depth = int(N // (self.grid_size * self.grid_size))
        if T is None or H_patches is None or W_patches is None:
            count = int(grid_depth * self.grid_size * self.grid_size)
        else:
            count = int(T * H_patches * W_patches)
        mask = torch.arange(count, device=x.device)
    d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
    if self.interpolate_rope:
        h_patches = int(self.grid_size) if H_patches is None else H_patches
        w_patches = int(self.grid_size) if W_patches is None else W_patches
        h_mask = h_mask * (self.pretrained_grid_size - 1) / (h_patches - 1)
        w_mask = w_mask * (self.pretrained_grid_size - 1) / (w_patches - 1)
    sin, cos = tables(self, [d_mask, h_mask, w_mask], q.dtype)
    q = (q * cos) + (_pairs(q) * sin)
    k = (k * cos) + (_pairs(k) * sin)
    with torch.backends.cuda.sdp_kernel():
        x = functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal
        )
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, None

"""The semantic predictor: VL-JEPA's scheme with a small predictor (§4.4.6).

The encoder's tokens of the whole clip (no mask) are projected to the predictor's width and
joined by learned queries; bidirectional attention with 3D RoPE on the video tokens runs over
both; the query outputs are averaged and projected to the caption space, giving ŷ, the
retrieval vector. With ``hypotheses = K > 1`` (ESP-4, optional) the queries form K groups and
each group's mean is one hypothesis ŷ_k, at zero extra parameters (§4.5.9).
"""

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .config import SemanticSettings
from .masking import TokenGrid
from .rope import RoPE3D


class DropPath(nn.Module):
    """Stochastic depth: drops a residual branch for a whole sample during training."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        dropped: Tensor = x * x.new_empty(shape).bernoulli_(keep) / keep
        return dropped


class Block(nn.Module):
    """Pre-norm transformer block with RoPE attention, LayerScale and stochastic depth."""

    def __init__(self, width: int, heads: int, settings: SemanticSettings, rope: RoPE3D) -> None:
        super().__init__()
        self.heads = heads
        self.rope = rope
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        hidden = int(width * settings.mlp_ratio)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)
        self.gamma1 = nn.Parameter(torch.full((width,), settings.layer_scale))
        self.gamma2 = nn.Parameter(torch.full((width,), settings.layer_scale))
        self.dropout = nn.Dropout(settings.dropout)
        self.drop_path = DropPath(settings.drop_path)

    def attention(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, tokens, width = x.shape
        q, k, v = self.qkv(x).view(batch, tokens, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k = RoPE3D.apply(q, cos, sin), RoPE3D.apply(k, cos, sin)
        p = self.dropout.p if self.training else 0.0
        out = functional.scaled_dot_product_attention(q, k, v, dropout_p=p)
        projected: Tensor = self.proj(out.transpose(1, 2).reshape(batch, tokens, width))
        return projected

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.drop_path(self.gamma1 * self.dropout(self.attention(self.norm1(x), cos, sin)))
        mlp = self.fc2(self.dropout(functional.gelu(self.fc1(self.norm2(x)))))
        output: Tensor = x + self.drop_path(self.gamma2 * self.dropout(mlp))
        return output


class SemanticPredictor(nn.Module):
    def __init__(self, input_dim: int, settings: SemanticSettings, grid: TokenGrid) -> None:
        super().__init__()
        self.settings = settings
        self.grid = grid
        width = settings.width
        self.rope = RoPE3D(width // settings.heads, grid)
        self.inputs = nn.Linear(input_dim, width)
        self.queries = nn.Parameter(torch.randn(settings.queries, width) * 0.02)
        self.blocks = nn.ModuleList(
            Block(width, settings.heads, settings, self.rope) for _ in range(settings.depth)
        )
        self.norm = nn.LayerNorm(width)
        self.outputs = nn.Linear(width, settings.output_dim)

    def _inputs(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if tokens.shape[1] != self.grid.size:
            raise ValueError(f"expected {self.grid.size} tokens, got {tokens.shape[1]}")
        cos, sin = self.rope.rotation(tokens.device)
        x = torch.cat([self.inputs(tokens), self.queries.expand(len(tokens), -1, -1)], dim=1)
        return x, cos.to(x.dtype), sin.to(x.dtype)

    def query_outputs(self, tokens: Tensor) -> Tensor:
        """(batch, queries, width): the normalised output of every query, before the mean."""
        x, cos, sin = self._inputs(tokens)
        for block in self.blocks:
            x = block(x, cos, sin)
        outputs: Tensor = self.norm(x[:, -self.settings.queries :])
        return outputs

    def project(self, queries: Tensor) -> Tensor:
        """Query outputs to (batch, K, output_dim): the mean of each group of queries."""
        groups = queries.view(len(queries), self.settings.hypotheses, -1, queries.shape[-1])
        predicted: Tensor = self.outputs(groups.mean(dim=2))
        return predicted

    def forward(self, tokens: Tensor) -> Tensor:
        """(batch, N, input_dim) encoder tokens of the whole clip to (batch, K, output_dim)."""
        return self.project(self.query_outputs(tokens))

    @torch.no_grad()
    def query_attention(self, tokens: Tensor) -> Tensor:
        """(batch, heads, queries, N): how the queries attend to the video in the last block."""
        x, cos, sin = self._inputs(tokens)
        for block in self.blocks[:-1]:
            x = block(x, cos, sin)
        last = cast(Block, self.blocks[-1])
        batch, length, width = x.shape
        qkv = last.qkv(last.norm1(x)).view(batch, length, 3, last.heads, -1).permute(2, 0, 3, 1, 4)
        q, k = RoPE3D.apply(qkv[0], cos, sin), RoPE3D.apply(qkv[1], cos, sin)
        queries = q[:, :, -self.settings.queries :]
        scores = queries @ k.transpose(-2, -1) / (width // last.heads) ** 0.5
        attention: Tensor = scores.softmax(dim=-1)[..., : self.grid.size]
        return attention

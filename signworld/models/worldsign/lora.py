"""Low-rank adaptation of frozen linear layers (§4.4.2).

``W' = W + (alpha / r) · B A``, with ``A`` Gaussian and ``B`` zero, so an adapted layer computes
exactly the pre-trained one at step 0. A fused projection such as ``qkv`` gets one adapter per
output slice (q, k, v), which is what the parameter counts of §4.8 assume.

``LoRALinear`` wraps a layer whose ``forward`` is called. ``LoRAWeight`` adapts a weight in
place, as a parametrization, for modules that read their weights directly: PyTorch's
``nn.MultiheadAttention`` (S-JEPA) passes ``in_proj_weight`` and ``out_proj.weight`` to a
functional call and never calls ``out_proj``.
"""

from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus ``splits`` independent low-rank adapters on its output."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, splits: int = 1) -> None:
        super().__init__()
        if base.out_features % splits:
            raise ValueError(f"{base.out_features} outputs do not split into {splits} parts")
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        width = base.out_features // splits
        self.down = nn.ParameterList(
            nn.Parameter(torch.randn(rank, base.in_features) / rank) for _ in range(splits)
        )
        self.up = nn.ParameterList(nn.Parameter(torch.zeros(width, rank)) for _ in range(splits))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: Tensor) -> Tensor:
        update = torch.cat([(x @ a.T) @ b.T for a, b in zip(self.down, self.up, strict=True)], -1)
        output: Tensor = self.base(x) + self.scale * update
        return output

    def adapter_parameters(self) -> list[nn.Parameter]:
        return [*self.down, *self.up]


class LoRAWeight(nn.Module):
    """Parametrization ``W ↦ W + (alpha / r) · B A`` of a frozen weight, one adapter per slice."""

    def __init__(self, weight: Tensor, rank: int, alpha: float, splits: int = 1) -> None:
        super().__init__()
        out_features, in_features = weight.shape
        if out_features % splits:
            raise ValueError(f"{out_features} outputs do not split into {splits} parts")
        self.scale = alpha / rank
        width = out_features // splits
        device = weight.device
        self.down = nn.ParameterList(
            nn.Parameter(torch.randn(rank, in_features, device=device) / rank)
            for _ in range(splits)
        )
        self.up = nn.ParameterList(
            nn.Parameter(torch.zeros(width, rank, device=device)) for _ in range(splits)
        )

    def forward(self, weight: Tensor) -> Tensor:
        delta = torch.cat([b @ a for a, b in zip(self.down, self.up, strict=True)], dim=0)
        return weight + self.scale * delta.to(weight.dtype)

    def adapter_parameters(self) -> list[nn.Parameter]:
        return [*self.down, *self.up]


def adapt_weight(
    module: nn.Module, name: str, rank: int, alpha: float, splits: int = 1
) -> LoRAWeight:
    """Freeze ``module.<name>`` and register a LoRA parametrization on it."""
    weight: Tensor = getattr(module, name)
    weight.requires_grad_(False)
    adapter = LoRAWeight(weight, rank, alpha, splits)
    parametrize.register_parametrization(module, name, adapter)
    return adapter


def inject(module: nn.Module, targets: Iterable[str], rank: int, alpha: float) -> list[LoRALinear]:
    """Replace every ``nn.Linear`` child named in ``targets`` below ``module`` by a LoRA layer.

    A layer named ``qkv`` gets three adapters. Returns the new layers, in module order.
    """
    wanted = set(targets)
    created: list[LoRALinear] = []
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if name in wanted and isinstance(child, nn.Linear):
                layer = LoRALinear(child, rank, alpha, splits=3 if name == "qkv" else 1)
                setattr(parent, name, layer.to(child.weight.device))
                created.append(layer)
    if not created:
        raise ValueError(f"no linear layer named {sorted(wanted)} under {type(module).__name__}")
    return created


def adapter_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Every LoRA parameter under ``module``."""
    return [
        p
        for layer in module.modules()
        if isinstance(layer, LoRALinear | LoRAWeight)
        for p in layer.adapter_parameters()
    ]

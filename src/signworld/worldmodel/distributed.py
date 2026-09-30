"""Data-parallel training across GPUs: the process group and the collectives the losses need.

WorldSign trains with PyTorch's DistributedDataParallel: every GPU holds the whole model and a
share of the batch, and the gradients of the trainable parameters (22 M, LoRA and new modules)
are averaged after each backward. Sharding (FSDP, ZeRO) would save only the model states,
about 2 GB here, while the memory goes to activations, and it would add an all-gather of the
frozen 330 M weights to every forward.

Three terms need the whole batch rather than one GPU's share: SIGReg (the characteristic
function of all the samples, and their number N), InfoNCE and L_unif (all the pairs). Each
rank computes them identically from collectives with a backward, and DDP's average of the
gradients then gives exactly the gradient of the term. For ``y = Σ_r x_r`` computed on every
rank, the backward of ``all_sum`` sums the ranks' identical gradients, so rank r receives
``W · ∂L/∂y``; DDP divides by W. ``all_gather`` does the same slice by slice. The per-sample
terms (E_fis, E_sem, L_anchor) stay local means, and DDP's average makes them global means.

Launch with ``torchrun --standalone --nproc_per_node=<gpus>``; one process runs without a
process group and every collective is the identity.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor


class _AllSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        total = x.clone()
        dist.all_reduce(total)
        return total

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> Tensor:
        total = grad.contiguous().clone()
        dist.all_reduce(total)
        return total


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        ctx.rank, ctx.rows = dist.get_rank(), x.shape[0]
        parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, x.contiguous())
        return torch.cat(parts)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> Tensor:
        total = grad.contiguous().clone()
        dist.all_reduce(total)
        return total[ctx.rank * ctx.rows : (ctx.rank + 1) * ctx.rows]


@dataclass(frozen=True, slots=True)
class Distributed:
    """This process's place in the run; with one process every collective is the identity."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = torch.device("cpu")  # noqa: RUF009 (immutable)

    @classmethod
    def from_environment(cls, backend: str | None = None) -> "Distributed":
        """Join the process group ``torchrun`` describes, or run alone without one."""
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        cuda = torch.cuda.is_available()
        device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
        if cuda:
            torch.cuda.set_device(device)
        if world_size == 1:
            return cls(device=device)
        if not dist.is_initialized():
            dist.init_process_group(
                backend or ("nccl" if cuda else "gloo"), device_id=device if cuda else None
            )
        return cls(dist.get_rank(), world_size, local_rank, device)

    @property
    def active(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def all_sum(self, x: Tensor) -> Tensor:
        """Σ over the ranks, with a backward (see the module docstring)."""
        return _AllSum.apply(x) if self.active else x  # type: ignore[no-untyped-call]

    def all_gather(self, x: Tensor) -> Tensor:
        """The ranks' tensors (same shape) concatenated along dim 0, with a backward."""
        return _AllGather.apply(x) if self.active else x  # type: ignore[no-untyped-call]

    def gather_objects(self, values: Sequence[Any]) -> list[Any]:
        """Every rank's ``values`` concatenated, in rank order (no gradient)."""
        if not self.active:
            return list(values)
        gathered: list[list[Any] | None] = [None] * self.world_size
        dist.all_gather_object(gathered, list(values))
        return [value for part in gathered for value in (part or [])]

    def mean(self, value: float) -> float:
        """The ranks' mean of a number, for logs (no gradient)."""
        if not self.active:
            return value
        tensor = torch.tensor(value, device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor)
        return float(tensor) / self.world_size

    def any(self, flag: bool) -> bool:
        """True if any rank's flag is: a stop decided by one rank stops them all."""
        if not self.active:
            return flag
        tensor = torch.tensor(int(flag), device=self.device)
        dist.all_reduce(tensor)
        return bool(tensor)

    def barrier(self) -> None:
        if self.active:
            dist.barrier()

    def shutdown(self) -> None:
        if self.active and dist.is_initialized():
            dist.destroy_process_group()


SINGLE = Distributed()
"""One process, no process group."""

"""Collaudo «Infrastruttura» (§4.13.1): two data-parallel processes equal one process.

Two CPU processes (gloo) each take half of a batch; with DDP they must produce the gradients,
and the batch-level terms (SIGReg per modality and per articulator, InfoNCE, L_unif), that one
process computes on the whole batch.
"""

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from signworld.worldmodel.config import LossSettings, SemanticSettings
from signworld.worldmodel.distributed import SINGLE, Distributed
from signworld.worldmodel.losses import Objective
from signworld.worldmodel.trainer import Trainer

from .conftest import needs_hub, slow, synthetic_corpus, tiny_teacher, tiny_training

WORLD = 2
BATCH = 16


class _Toy(nn.Module):
    """Video, text and pose heads over fixed inputs, with the objective inside, as WorldSign."""

    def __init__(self, arm: str, collective: Distributed) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.video = nn.Linear(4, 8)
        self.text = nn.Linear(6, 8)
        self.pose = nn.Linear(3, 8)
        settings = LossSettings(arm=arm, sigreg_directions=64)  # type: ignore[arg-type]
        self.objective = Objective(settings, SemanticSettings(), True, collective)

    def forward(self, data: dict[str, Tensor], videos: list[str]) -> dict[str, Tensor]:
        predicted = self.video(data["x"])[:, None]
        terms = self.objective.semantic_terms(
            predicted, self.text(data["t"]), videos, torch.Generator().manual_seed(7)
        )
        latent = self.pose(data["p"])
        terms["sigreg_posa"] = self.objective.pose_sigreg(
            latent, data["present"], torch.Generator().manual_seed(8)
        )
        return terms


def _data() -> tuple[dict[str, Tensor], list[str]]:
    generator = torch.Generator().manual_seed(1)
    present = torch.rand(BATCH, 5, 4, generator=generator) > 0.3
    present[BATCH // 2 :, :, 2] = False  # the second process never sees articulator 2
    present[:, :, 3] = False  # articulator 3 is absent from the whole batch
    present[0, 0, 1] = True
    data = {
        "x": torch.randn(BATCH, 4, generator=generator),
        "t": torch.randn(BATCH, 6, generator=generator),
        "p": torch.randn(BATCH, 5, 4, 3, generator=generator),
        "present": present,
    }
    videos = [f"v{i // 3}" for i in range(BATCH)]  # clips of one video straddle the halves
    return data, videos


def _step(model: nn.Module, data: dict[str, Tensor], videos: list[str]) -> dict[str, Tensor]:
    terms: dict[str, Tensor] = model(data, videos)
    sum(terms.values()).backward()  # type: ignore[union-attr]
    return terms


def _worker(rank: int, directory: str, arm: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/store", rank=rank, world_size=WORLD
    )
    collective = Distributed(rank, WORLD, rank, torch.device("cpu"))
    model = DistributedDataParallel(_Toy(arm, collective))
    data, videos = _data()
    share = slice(rank * BATCH // WORLD, (rank + 1) * BATCH // WORLD)
    terms = _step(model, {k: v[share] for k, v in data.items()}, videos[share])
    grads = {name: p.grad for name, p in model.module.named_parameters()}
    batch_terms = {k: v.detach() for k, v in terms.items() if k != "e_sem"}
    torch.save({"grads": grads, "terms": batch_terms}, Path(directory) / f"rank{rank}.pt")
    dist.destroy_process_group()


@pytest.mark.parametrize("arm", ["B", "C"])
def test_two_processes_give_the_gradients_and_terms_of_one(tmp_path: Path, arm: str) -> None:
    mp.spawn(_worker, args=(str(tmp_path), arm), nprocs=WORLD, join=True)
    reference = _Toy(arm, SINGLE)
    terms = _step(reference, *_data())

    for rank in range(WORLD):
        saved = torch.load(tmp_path / f"rank{rank}.pt")
        for name, parameter in reference.named_parameters():
            assert parameter.grad is not None
            assert torch.allclose(saved["grads"][name], parameter.grad, atol=1e-5), name
        for key, value in saved["terms"].items():
            assert torch.allclose(value, terms[key].detach(), atol=1e-5), key
    assert set(terms) >= {"sigreg_sem", "sigreg_posa"}


def _train_worker(rank: int, directory: str, corpus: Any) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/train-store", rank=rank, world_size=WORLD
    )
    collective = Distributed(rank, WORLD, rank, torch.device("cpu"))
    root = Path(directory)
    model, config, train, validation = tiny_training(
        root, corpus, teacher=root / "sjepa.pt", collective=collective
    )
    state = Trainer(model, config, train, validation, root / "run", collective).fit()
    trained = {n: p.detach().clone() for n, p in model.named_parameters()}
    torch.save({"step": state.step, "params": trained}, root / f"trained{rank}.pt")
    dist.destroy_process_group()


@slow
@needs_hub
def test_a_ddp_run_keeps_every_gpu_identical(tmp_path: Path) -> None:
    """P-Infrastruttura: parameters identical on every GPU after the run, curriculum included."""
    corpus = synthetic_corpus(tmp_path / "corpus")
    tiny_teacher(tmp_path)

    mp.spawn(_train_worker, args=(str(tmp_path), corpus), nprocs=WORLD, join=True)

    runs = [torch.load(tmp_path / f"trained{rank}.pt") for rank in range(WORLD)]
    assert runs[0]["step"] == runs[1]["step"] > 0
    for name, value in runs[0]["params"].items():
        assert torch.equal(value, runs[1]["params"][name]), name
    assert (tmp_path / "run" / "checkpoints" / "final.pt").is_file()

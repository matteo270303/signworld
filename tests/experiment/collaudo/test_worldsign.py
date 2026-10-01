"""The collaudo tools of the training system on tiny real modules (§4.13.1)."""

import math
from pathlib import Path

from signworld.worldmodel.collaudo import measure_efficiency, overfit
from signworld.worldmodel.data import Collate
from signworld.worldmodel.trainer import Trainer

from .conftest import needs_hub, slow, synthetic_corpus, tiny_training


@slow
@needs_hub
def test_overfit_trains_and_reads_every_criterion(tmp_path: Path) -> None:
    corpus = synthetic_corpus(tmp_path / "corpus")
    model, _, train, _, _ = tiny_training(tmp_path, corpus)
    batch = Collate(model.text.centering.languages)([train[(i, 0)] for i in range(4)])

    result = overfit(model, batch, steps=3)

    assert result.steps == 3
    assert 0.0 <= result.r1 <= 1.0 and not math.isnan(result.e_sem)
    assert not math.isnan(result.r2_visible)


@slow
@needs_hub
def test_efficiency_counts_flops_and_times_steps(tmp_path: Path) -> None:
    corpus = synthetic_corpus(tmp_path / "corpus")
    model, config, train, validation, probe = tiny_training(tmp_path, corpus)
    trainer = Trainer(model, config, train, validation, tmp_path / "run", probe_data=probe)

    result = measure_efficiency(trainer, steps=2, warmup=1, peak_flops=1e12)

    assert result.loader_clips_per_second > 0 and result.step_seconds > 0
    assert result.flops_per_step > 1e6
    assert 0 < result.utilisation < 1

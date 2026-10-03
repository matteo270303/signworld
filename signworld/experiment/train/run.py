"""A training run from its configuration: model, statistics, data and trainer.

The statistics fitted on the training clips (the caption means per language and the keypoint
scale of L_anchor) are computed once by the first GPU, saved next to the checkpoints and read
by every GPU, so all start from the same numbers (§4.6: nothing from validation).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch

from signworld.data.loaders import (
    TRAIN,
    ClipDataset,
    Collate,
    caption_statistics,
    keypoint_statistics,
    read_index,
    validation_subset,
)
from signworld.data.text import EmbeddingStore
from signworld.models.worldsign.model import WorldSign, build_worldsign

from .checkpoint import TrainingState
from .config import WorldSignConfig
from .distributed import Distributed
from .preflight import enforce, run_preflight
from .trainer import Trainer

logger = logging.getLogger(__name__)

STATISTICS = "statistics.pt"


def fit_statistics(
    model: WorldSign, train: pa.Table, embeddings: np.ndarray, clips: int
) -> dict[str, Any]:
    """The centring means and the keypoint variance, fitted on training clips only."""
    captions, languages = caption_statistics(train, embeddings)
    model.text.centering.fit(captions, languages)
    if model.pose is not None:
        model.pose.fit(*keypoint_statistics(train, clips))
    return {
        "centering": model.text.centering.state_dict(),
        "keypoint_variance": None if model.pose is None else model.pose.keypoint_variance,
    }


def load_statistics(model: WorldSign, saved: dict[str, Any]) -> None:
    model.text.centering.load_state_dict(saved["centering"])
    if model.pose is not None and saved["keypoint_variance"] is not None:
        model.pose.keypoint_variance.copy_(saved["keypoint_variance"])


@dataclass(frozen=True, slots=True)
class Prepared:
    """A run ready to start: its trainer and what the assertions before launching read."""

    trainer: Trainer
    train: pa.Table
    train_data: ClipDataset
    embeddings: np.ndarray


def prepare(config: WorldSignConfig, output: Path, collective: Distributed) -> Prepared:
    """The model with its statistics, the datasets and the trainer of a run."""
    data = config.data
    if data.index is None or data.embeddings is None:
        raise ValueError("data.index and data.embeddings must name the stage-0 outputs")
    model = build_worldsign(config, collective=collective)
    embeddings = EmbeddingStore(data.embeddings).embeddings()
    train = read_index(data.index, TRAIN)
    statistics = output / STATISTICS
    if collective.is_main and not statistics.is_file():
        output.mkdir(parents=True, exist_ok=True)
        torch.save(fit_statistics(model, train, embeddings, data.statistics_clips), statistics)
    collective.barrier()
    load_statistics(model, torch.load(statistics, map_location="cpu", weights_only=False))

    known = pa.array(list(model.text.centering.languages))
    held_out = read_index(data.index, data.validation_split)
    held_out = held_out.filter(pc.is_in(held_out.column("caption_language"), known))
    validation = validation_subset(held_out, data.validation_clips)
    logger.info("%d training clips, %d validation clips", train.num_rows, validation.num_rows)
    threshold = config.physical.box_threshold
    train_data = ClipDataset(
        train,
        embeddings,
        augmentation=config.augmentation if config.augmentation.enabled else None,
        box_threshold=threshold,
        seed=config.training.seed,
    )
    trainer = Trainer(
        model,
        config,
        train_data,
        ClipDataset(validation, embeddings, augmentation=None, box_threshold=threshold),
        output,
        collective,
        probe_data=ClipDataset(
            validation_subset(train, config.diagnostics.probe_clips),
            embeddings,
            augmentation=None,
            box_threshold=threshold,
        ),
        train_subset_data=ClipDataset(
            validation_subset(train, validation.num_rows),
            embeddings,
            augmentation=None,
            box_threshold=threshold,
        ),
    )
    return Prepared(trainer, train, train_data, embeddings)


def run(config: WorldSignConfig, output: Path, collective: Distributed) -> TrainingState:
    """The assertions before launching (§4.13.2), then the training loop."""
    prepared = prepare(config, output, collective)
    trainer, model = prepared.trainer, prepared.trainer.model
    fresh = not trainer.checkpoints.exists("latest")
    data = config.data
    assert data.index is not None  # checked by prepare
    index = read_index(data.index)
    names = sorted(set(index.column("split").to_pylist()))
    splits = {name: read_index(data.index, name) for name in names}
    batch = (
        Collate(model.text.centering.languages)(
            [prepared.train_data[(index, 0)] for index in range(trainer.per_gpu)]
        )
        .to(collective.device)
        .augmented()
    )
    report = run_preflight(
        model,
        config,
        splits,
        batch,
        caption_statistics(prepared.train, prepared.embeddings),
        output / "checksums.json",
        collective,
        overfit_steps=config.training.preflight_overfit_steps if fresh else 0,
        checksum_files=_checksum_files(config),
    )
    if collective.is_main:
        report.write(output / "preflight.json")
        (output / "config.json").write_text(config.model_dump_json(indent=2) + "\n")
    enforce(report)
    return trainer.fit()


def _checksum_files(config: WorldSignConfig) -> dict[str, Path]:
    """The pre-trained files a run reads: V-JEPA 2.1 and the caption embeddings (the pose
    encoder is trained from scratch)."""
    files = {"encoder": config.encoder.checkpoint}
    if config.data.embeddings is not None:
        files["embeddings"] = config.data.embeddings / "embeddings.npy"
    return files

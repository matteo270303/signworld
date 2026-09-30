"""A training run from its configuration: model, statistics, data and trainer.

The statistics fitted on the training clips (the caption means per language and the keypoint
scale of L_anchor) are computed once by the first GPU, saved next to the checkpoints and read
by every GPU, so all start from the same numbers (§4.6: nothing from validation).
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch

from ..text.embedding import EmbeddingStore
from .checkpoint import TrainingState
from .config import WorldSignConfig
from .data import (
    SPLITS,
    TRAIN,
    VALIDATION_CHANNEL,
    ClipDataset,
    Collate,
    caption_statistics,
    keypoint_statistics,
    read_index,
    validation_subset,
)
from .distributed import Distributed
from .model import WorldSign, build_worldsign
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


def run(config: WorldSignConfig, output: Path, collective: Distributed) -> TrainingState:
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
    held_out = read_index(data.index, VALIDATION_CHANNEL)
    held_out = held_out.filter(pc.is_in(held_out.column("caption_language"), known))
    validation = validation_subset(held_out, data.validation_clips)
    logger.info("%d training clips, %d validation clips", train.num_rows, validation.num_rows)
    train_data = ClipDataset(
        train,
        embeddings,
        augmentation=config.augmentation,
        box_threshold=config.physical.box_threshold,
        seed=config.training.seed,
    )
    trainer = Trainer(
        model,
        config,
        train_data,
        ClipDataset(
            validation, embeddings, augmentation=None, box_threshold=config.physical.box_threshold
        ),
        output,
        collective,
    )
    fresh = not trainer.checkpoints.exists("latest")
    splits = {name: read_index(data.index, name) for name in SPLITS}
    batch = Collate(model.text.centering.languages)(
        [train_data[(index, 0)] for index in range(trainer.per_gpu)]
    ).to(collective.device)
    report = run_preflight(
        model,
        config,
        splits,
        batch,
        caption_statistics(train, embeddings),
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
    files = {"encoder": config.encoder.checkpoint}
    if config.physical.enabled and config.pose_encoder.checkpoint is not None:
        files["pose_encoder"] = config.pose_encoder.checkpoint
    if config.data.embeddings is not None:
        files["embeddings"] = config.data.embeddings / "embeddings.npy"
    return files

"""PC2's ridge baseline on the run's own splits: the floor every arm has to beat (§4.12.1).

The frozen V-JEPA 2.1 encoder reads the clips exactly as training does (``ClipDataset``, no
augmentation); the mean of its last-layer tokens is each clip's feature, as in PC2. A ridge
from features to the caption embeddings (whole, unit length, centred per language on the
training clips) is fitted on a fixed subset of training clips; its penalty is the one with the
best mean R@1 of the two directions on the validation subset the training validates on, and it
is then read on every held-out split, each a gallery of its own. The report gives the R@1 of
the validation split in the form ``diagnostics.ridge_baseline`` takes (fractions), for the
gate and stop F3, and every split's measures with bootstrap intervals, for the paper's tables.
"""

import datetime
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore[import-untyped]

from signworld.data.loaders import TRAIN, ClipDataset, Collate, read_index, validation_subset
from signworld.data.text import EmbeddingStore
from signworld.experiment.studies.video_probes import TextScores, TextTargets, split_text_scores
from signworld.experiment.train.config import WorldSignConfig
from signworld.models.encoders.video_encoders import load_vjepa2_1, model_input


@torch.no_grad()
def mean_tokens(  # noqa: PLR0913 (the encoder, the clips and how to read them)
    encoder: nn.Module,
    table: pa.Table,
    embeddings: np.ndarray,
    device: torch.device,
    *,
    batch: int = 16,
    workers: int = 8,
    box_threshold: float = 0.3,
) -> np.ndarray:
    """(clips, width) the mean last-layer token of every clip of ``table``, in its order."""
    dataset = ClipDataset(table, embeddings, augmentation=None, box_threshold=box_threshold)
    languages = sorted({str(v) for v in table.column("caption_language").to_pylist()})
    loader: DataLoader[Any] = DataLoader(
        dataset,
        batch_size=batch,
        sampler=[(index, 0) for index in range(len(dataset))],
        num_workers=workers,
        collate_fn=Collate(languages),
    )
    features = []
    bf16 = device.type == "cuda"
    for clips in tqdm(loader, desc="[ridge features]", mininterval=10.0, leave=False):
        frames = clips.frames.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            tokens = encoder(model_input(frames))
        features.append(tokens.float().mean(1).cpu().numpy())
    return np.concatenate(features)


def _targets(table: pa.Table, embeddings: np.ndarray, train: pa.Table) -> TextTargets:
    """The caption of every clip, centred per caption language on the training clips, the
    languages the model's own centring uses."""
    rows = np.asarray(table.column("caption_row").to_pylist())
    languages = np.asarray(table.column("caption_language").to_pylist(), dtype=object)
    fitting_rows = np.asarray(train.column("caption_row").to_pylist())
    fitting_languages = np.asarray(train.column("caption_language").to_pylist(), dtype=object)
    joined = TextTargets.of(
        np.concatenate([fitting_rows, rows]),
        embeddings,
        np.concatenate([fitting_languages, languages]),
        np.concatenate([np.ones(len(fitting_rows), bool), np.zeros(len(rows), bool)]),
    )
    return TextTargets(joined.embeddings[len(fitting_rows) :], joined.groups[len(fitting_rows) :])


def ridge_baseline(  # noqa: PLR0913 (the run's settings and the sizes of the subsets)
    config: WorldSignConfig,
    splits: Sequence[str],
    device: torch.device,
    *,
    fit_clips: int = 20_000,
    clips: int | None = None,
    batch: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """The baseline of every split in ``splits`` (the first chooses the penalty).

    The validation split takes ``data.validation_clips`` clips, the subset training validates
    on; the others take ``clips`` (all when None), always the first ones by clip ID."""
    data = config.data
    if data.index is None or data.embeddings is None:
        raise ValueError("data.index and data.embeddings must name the stage-0 outputs")
    embeddings = np.asarray(EmbeddingStore(data.embeddings).embeddings())
    train = read_index(data.index, TRAIN)
    known = pa.array(sorted({str(v) for v in train.column("caption_language").to_pylist()}))
    fitting = validation_subset(train, fit_clips)
    held: dict[str, pa.Table] = {}
    for name in splits:
        table = read_index(data.index, name)
        table = table.filter(pc.is_in(table.column("caption_language"), known))
        size = data.validation_clips if name == data.validation_split else clips
        held[name] = table if size is None else validation_subset(table, size)
    encoder, _, _ = load_vjepa2_1(
        config.encoder.hub_repo, config.encoder.entrypoint, config.encoder.checkpoint
    )
    encoder = encoder.eval().to(device)
    threshold = config.physical.box_threshold

    def features(table: pa.Table) -> np.ndarray:
        return mean_tokens(encoder, table, embeddings, device, batch=batch, box_threshold=threshold)

    scores = split_text_scores(
        features(fitting),
        _targets(fitting, embeddings, fitting),
        {name: (features(t), _targets(t, embeddings, fitting)) for name, t in held.items()},
        choose_on=splits[0],
        seed=seed,
    )
    first = scores[splits[0]]
    return {
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "encoder": str(config.encoder.checkpoint),
        "index": str(data.index),
        "fit_clips": len(fitting),
        "seed": seed,
        "penalty_chosen_on": splits[0],
        "ridge_baseline": {"t2v": first.t2v_r1.estimate, "v2t": first.v2t_r1.estimate},
        "splits": {name: _as_dict(value) for name, value in scores.items()},
    }


def _as_dict(scores: TextScores) -> dict[str, Any]:
    return {
        "gallery": scores.gallery,
        "penalty": scores.penalty,
        "t2v_r1": asdict(scores.t2v_r1),
        "v2t_r1": asdict(scores.v2t_r1),
        "measures": scores.measures,
    }


def write_ridge_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path

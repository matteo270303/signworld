"""Tiny but real V-JEPA 2.1 modules, built from Meta's code with a few channels and frames."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pytest
import torch
from torch import nn

from signworld.corpus.manifest import SCHEMA as MANIFEST_SCHEMA
from signworld.corpus.materialize import MaterializedRecord
from signworld.models.pose_teachers import SJEPATeacher, TeacherShape
from signworld.pose.frames import FRAMES_PER_CLIP as FRAMES
from signworld.pose.tokens import JOINTS, STEPS
from signworld.worldmodel.config import DataSettings, load_config
from signworld.worldmodel.data import (
    TRAIN,
    VALIDATION_CHANNEL,
    ClipDataset,
    build_training_index,
    validation_subset,
)
from signworld.worldmodel.distributed import SINGLE, Distributed
from signworld.worldmodel.masking import TokenGrid
from signworld.worldmodel.model import WorldSign, assemble_worldsign
from signworld.worldmodel.pose_branch import PoseBranch
from signworld.worldmodel.run import fit_statistics
from signworld.worldmodel.video_branch import assemble

HUB = Path("/lustrehome/mvigone/cache/torch/hub/facebookresearch_vjepa2_main")
BASE = Path(__file__).resolve().parents[2] / "configs" / "model" / "worldsign.yaml"
ABLATIONS = BASE.parent / "ablations"
GRID = TokenGrid(steps=2, rows=4, columns=4)
"""4 frames of 64² in tubelets of 2 and patches of 16."""

needs_hub = pytest.mark.skipif(not HUB.is_dir(), reason="V-JEPA 2.1 hub code not available")
slow = pytest.mark.skipif(
    os.environ.get("SIGNWORLD_SLOW_TESTS") != "1",
    reason="minutes on CPU; run with SIGNWORLD_SLOW_TESTS=1",
)


def meta_modules(frames: int = 4) -> tuple[Any, Any]:
    """A 12-block encoder (so its hierarchical layers exist) and a 4-block predictor."""
    # Meta's repository first on the path only while importing: it has its own ``tests``
    # package, which would shadow ours in the processes the DDP tests spawn.
    sys.path.insert(0, str(HUB))
    try:
        from app.vjepa_2_1.models import predictor as vit_predictor  # noqa: PLC0415
        from app.vjepa_2_1.models import vision_transformer as vit_encoder  # noqa: PLC0415
    finally:
        sys.path.remove(str(HUB))

    torch.manual_seed(0)
    encoder = vit_encoder.VisionTransformer(
        img_size=(64, 64),
        patch_size=16,
        num_frames=frames,
        tubelet_size=2,
        embed_dim=48,
        depth=12,
        num_heads=4,
        use_rope=True,
        interpolate_rope=True,
        n_output_distillation=1,
        img_temporal_dim_size=1,
    )
    predictor = vit_predictor.vit_predictor(
        img_size=(64, 64),
        patch_size=16,
        num_frames=frames,
        tubelet_size=2,
        embed_dim=48,
        predictor_embed_dim=24,
        depth=4,
        num_heads=2,
        num_mask_tokens=2,
        use_mask_tokens=True,
        use_rope=True,
        n_output_distillation=1,
        return_all_tokens=True,
        teacher_embed_dim=40,
        img_temporal_dim_size=1,
    )
    return encoder, predictor


def tiny_config(*overlays: Path, **changes: Any) -> Any:
    """The project's YAML with the sizes of the tiny modules."""
    config = load_config(BASE, *overlays)
    encoder = config.encoder.model_copy(update={"levels": (2, 5, 8, 11)})
    semantic = config.semantic.model_copy(
        update={"width": 24, "heads": 2, "dropout": 0.0, "drop_path": 0.0}
    )
    physical = config.physical.model_copy(update={"target_dim": 8})
    pose = config.pose_encoder.model_copy(
        update={"width": POSE_SHAPE.width, "depth": POSE_SHAPE.depth, "heads": POSE_SHAPE.heads}
    )
    update = {
        "encoder": encoder,
        "semantic": semantic,
        "physical": physical,
        "pose_encoder": pose,
        **changes,
    }
    return config.model_copy(update=update)


POSE_SHAPE = TeacherShape(width=8, depth=2, heads=2, predictor_depth=1)
"""A tiny S-JEPA: the width matches the tiny physical head."""


def tiny_teacher(directory: Path) -> tuple[Path, SJEPATeacher]:
    """A tiny S-JEPA teacher saved as PC5 saves it, with EMA weights unlike the online ones."""
    torch.manual_seed(1)
    teacher = SJEPATeacher(POSE_SHAPE)
    with torch.no_grad():
        for parameter in teacher.target_encoder.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    path = directory / "sjepa.pt"
    torch.save(teacher.state_dict(), path)
    return path, teacher


def pose_tokens(batch: int, seed: int = 0) -> torch.Tensor:
    """(batch, 32, 69, 6) tokens: x, y and presence of two frames, a few joints missing."""
    generator = torch.Generator().manual_seed(seed)
    xy = torch.randn(batch, STEPS, len(JOINTS), 2, 2, generator=generator)
    present = (torch.rand(batch, STEPS, len(JOINTS), 2, 1, generator=generator) > 0.1).float()
    return torch.cat([xy * present, present], dim=-1).flatten(-2)


class _Attention(nn.Module):
    """The shapes of Meta's RoPE attention, and the attributes set_rope_grid looks for."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.grid_size = 24

    def separate_positions(self) -> None: ...


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = _Attention(width)
        self.mlp = nn.ModuleDict(
            {"fc1": nn.Linear(width, 4 * width), "fc2": nn.Linear(4 * width, width)}
        )


def full_size_stubs() -> tuple[nn.Module, nn.Module]:
    """ViT-L (24 x 1,024) and its predictor (12 x 384) with Meta's layer shapes, on meta."""
    encoder = nn.Module()
    encoder.blocks = nn.ModuleList(_Block(1024) for _ in range(24))
    encoder.norms_block = nn.ModuleList(nn.LayerNorm(1024) for _ in range(4))
    encoder.hierarchical_layers = [5, 11, 17, 23]
    encoder.embed_dim = 1024
    predictor = nn.Module()
    predictor.predictor_blocks = nn.ModuleList(_Block(384) for _ in range(12))
    predictor.predictor_embed = nn.Linear(1024, 384)
    predictor.predictor_norm = nn.LayerNorm(384)
    predictor.predictor_proj = nn.Linear(384, 1664)
    predictor.predictor_proj_context = nn.Linear(384, 1664)
    return encoder, predictor


@dataclass(frozen=True)
class SyntheticCorpus:
    """Materialised clips in the pipeline's formats: manifest, index records, captions."""

    manifest: pa.Table
    records: list[MaterializedRecord]
    caption_rows: pa.Table
    embeddings: np.ndarray


def synthetic_corpus(directory: Path, clips: int = 12, size: int = 64) -> SyntheticCorpus:
    """``clips`` clips of 64 frames over three channels and two sign languages.

    Every clip shows a bright square drifting across a dark frame; its pose puts the shoulders
    in place and every other keypoint at random, all with a high score.
    """
    rng = np.random.default_rng(0)
    rows, records, clip_rows = [], [], []
    for number in range(clips):
        clip_id = f"clip{number:02d}"
        video = directory / "clips" / f"{clip_id}.mp4"
        pose = directory / "poses" / f"{clip_id}.npz"
        video.parent.mkdir(parents=True, exist_ok=True)
        pose.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 25.0, (size, size))
        for frame in range(FRAMES):
            image = np.zeros((size, size, 3), dtype=np.uint8)
            left = (frame + 3 * number) % (size - 16)
            image[20:36, left : left + 16] = 200
            writer.write(image)
        writer.release()
        keypoints = rng.uniform(0.2, 0.8, (FRAMES, 133, 2))
        keypoints[:, 5] = (0.6, 0.45)  # left shoulder
        keypoints[:, 6] = (0.4, 0.45)  # right shoulder
        scores = np.full((FRAMES, 133, 1), 5.0)
        np.savez_compressed(
            pose,
            pose=np.concatenate([keypoints, scores], axis=-1).astype(np.float16),
            idx=np.arange(FRAMES, dtype=np.int32),
            total=np.int32(FRAMES),
        )
        channel = f"channel{number % 3}"
        rows.append(
            {
                "clip_id": clip_id,
                "video_id": f"video{number // 2}",
                "start_s": 0.0,
                "end_s": 2.56,
                "caption": f"caption {number // 2}",
                "caption_language": "en" if number % 2 else "es",
                "sign_language": "ase" if number % 3 != 2 else "lsf",
                "channel_id": channel,
                "split": None,
                "video_available": True,
            }
        )
        records.append(
            MaterializedRecord(clip_id, str(video), str(pose), 25.0, 0, FRAMES, [0, 0, 64, 64])
        )
        clip_rows.append({"clip_id": clip_id, "row": number // 2})
    embeddings = rng.normal(size=(clips // 2 + 1, 768)).astype(np.float32)
    return SyntheticCorpus(
        pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA),
        records,
        pa.Table.from_pylist(clip_rows),
        embeddings,
    )


MODEL_GRID = TokenGrid(steps=STEPS, rows=4, columns=4)
"""64 frames of 64² in tubelets of 2: as many steps as the pose has."""


def tiny_worldsign(
    directory: Path,
    *overlays: Path,
    teacher: Path | None = None,
    collective: Distributed = SINGLE,
    **changes: Any,
) -> tuple[WorldSign, Any]:
    """The whole model on tiny real modules, statistics not yet fitted."""
    checkpoint = teacher if teacher is not None else tiny_teacher(directory)[0]
    config = tiny_config(*overlays, **changes)
    pose_settings = config.pose_encoder.model_copy(update={"checkpoint": checkpoint})
    config = config.model_copy(update={"pose_encoder": pose_settings})
    encoder, predictor = meta_modules(frames=2 * STEPS)
    video = assemble(encoder, predictor, config, MODEL_GRID)
    pose = PoseBranch.from_settings(config.pose_encoder) if config.physical.enabled else None
    return assemble_worldsign(video, pose, config, collective), config


STAGES = {"1a": 1 / 6, "1": 1 / 6, "2a": 1 / 6}
"""Six steps: one each for 1a, 1 and 2a, then stage 2."""


def tiny_training(
    directory: Path,
    corpus: SyntheticCorpus,
    *,
    teacher: Path | None = None,
    collective: Distributed = SINGLE,
) -> tuple[Any, ...]:
    """A tiny model with its statistics fitted, a short run's settings and its datasets.

    Every diagnostic runs at a short cadence, so the whole fail-fast system is exercised.
    """
    model, config = tiny_worldsign(directory, teacher=teacher, collective=collective)
    training = config.training.model_copy(
        update={
            "batch_size": 4,
            "epochs": 3,
            "validation_every": 1000,
            "log_every": 1,
            "stages": STAGES,
            "warmup_fraction": 0.2,
            "cooldown_fraction": 0.2,
        }
    )
    data = config.data.model_copy(update={"workers": 0, "validation_fraction": 0.5})
    diagnostics = config.diagnostics.model_copy(
        update={
            "frequent_every": 2,
            "rare_every": 3,
            "probe_clips": 4,
            "diagnostic_clips": 4,
            "small_clips": 2,
            "order_clips": 2,
            "plausibility_clips": 2,
            "plausibility_masks": 2,
        }
    )
    config = config.model_copy(
        update={"training": training, "data": data, "diagnostics": diagnostics}
    )
    index = build_training_index(
        corpus.manifest, corpus.records, corpus.caption_rows, DataSettings(validation_fraction=0.5)
    )
    train = index.filter(index.column("split").to_numpy(zero_copy_only=False) == TRAIN)
    held_out = index.filter(
        index.column("split").to_numpy(zero_copy_only=False) == VALIDATION_CHANNEL
    )
    fit_statistics(model, train, corpus.embeddings, clips=4)
    train_data = ClipDataset(
        train, corpus.embeddings, augmentation=config.augmentation, box_threshold=0.3
    )
    validation = ClipDataset(
        validation_subset(held_out, 4), corpus.embeddings, augmentation=None, box_threshold=0.3
    )
    probe = ClipDataset(train, corpus.embeddings, augmentation=None, box_threshold=0.3)
    return model, config, train_data, validation, probe

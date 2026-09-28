"""Typed configuration of the manifests, data checks and caption embeddings."""

from pathlib import Path
from typing import Self

import yaml
from pydantic import Field, PositiveFloat, PositiveInt

from .acquisition.config import AcquisitionConfig, FrozenModel
from .checks.checkpoint import CheckpointSource
from .checks.pose_quality import OUTSIDE_LIMITS
from .checks.text_geometry import TextGeometrySettings
from .corpus.text import NEAR_DUPLICATE_RATIO
from .text.embedding import EmbeddingSettings


class ContaminationSettings(FrozenModel):
    benchmark: str = "openasl"
    margin_s: float = 2.0
    near_duplicate_ratio: float = NEAR_DUPLICATE_RATIO
    duration_tolerance_s: PositiveFloat = 0.1


class PoseCheckSettings(FrozenModel):
    outside_limits: dict[str, float | None] = Field(default_factory=lambda: dict(OUTSIDE_LIMITS))
    """Per articulator, the largest share of boxes beyond the frame A4 accepts; null: report."""
    selection_clips: PositiveInt = 200
    alignment_clips: PositiveInt = 50
    contact_sheets: PositiveInt = 50
    workers: PositiveInt = 16
    seed: int = 0


class TestDataSettings(FrozenModel):
    root: Path
    """Where derived test material goes; the downloaded datasets are never written to."""
    clips: PositiveInt = 2000
    min_duration_s: PositiveFloat = 1.0
    max_duration_s: PositiveFloat = 20.0
    crop_margin: float = 0.15
    size: PositiveInt = 256
    detection_frames: PositiveInt = 8
    seed: int = 0


class PoseTeacherSettings(FrozenModel):
    """The pose-teacher comparison (§4.4.3): model shape, pre-training and read-outs."""

    min_score: PositiveFloat = 1.0
    width: PositiveInt = 256
    depth: PositiveInt = 8
    heads: PositiveInt = 8
    predictor_depth: PositiveInt = 5
    decoder_depth: PositiveInt = 3
    mask_ratio: float = 0.9
    epochs: PositiveInt = 100
    batch_size: PositiveInt = 64
    learning_rate: PositiveFloat = 1e-3
    min_learning_rate: PositiveFloat = 1e-5
    warmup_fraction: float = 0.05
    weight_decay: float = 0.05
    ema_start: float = 0.999
    centre_rate: float = 0.9
    probe_clips: PositiveInt = 8000
    """Clips used by the linear read-outs; the teachers train on every training clip."""
    seed: int = 0


class AnalysisConfig(FrozenModel):
    acquisition_config: Path
    """The acquisition settings, which define where every dataset lives."""
    models_root: Path
    """Where pre-trained checkpoints are downloaded, with their inspection reports."""
    threads: PositiveInt = 8
    checkpoints: dict[str, CheckpointSource] = Field(default_factory=dict)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    text_geometry: TextGeometrySettings = Field(default_factory=TextGeometrySettings)
    contamination: ContaminationSettings = Field(default_factory=ContaminationSettings)
    pose_checks: PoseCheckSettings = Field(default_factory=PoseCheckSettings)
    pose_teachers: PoseTeacherSettings = Field(default_factory=PoseTeacherSettings)
    test_data: TestDataSettings

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        with path.open(encoding="utf-8") as stream:
            return cls.model_validate(yaml.safe_load(stream))

    def acquisition(self) -> AcquisitionConfig:
        return AcquisitionConfig.from_yaml(self.acquisition_config)

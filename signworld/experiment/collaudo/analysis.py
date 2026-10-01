"""Typed configuration of the manifests, data checks and caption embeddings."""

from pathlib import Path
from typing import Literal, Self

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


class PoseIsotropySettings(FrozenModel):
    """Maps that make the S-JEPA latent isotropic, compared before and after (§4.4.3)."""

    whitening_shrinkage: tuple[float, ...] = (0.0, 1e-3, 1e-2)
    """ε of ``(Σ + ε·mean(λ) I)^(-1/2)``, in units of the mean eigenvalue."""
    iterations: tuple[PositiveInt, ...] = (1, 5, 10, 20, 50)
    """Iterations of RBIG and SINF at which the curve is read."""
    marginal_knots: PositiveInt = 1024
    sinf_directions: PositiveInt = 32
    """Directions Gaussianized per SINF iteration, out of the latent's 256."""
    sinf_steps: PositiveInt = 100
    sinf_rows: PositiveInt = 16384
    flow_depths: tuple[PositiveInt, ...] = (4, 8)
    flow_hidden: PositiveInt = 512
    flow_batch: PositiveInt = 4096
    flow_learning_rate: PositiveFloat = 1e-3
    flow_max_epochs: PositiveInt = 60
    flow_patience: PositiveInt = 5
    knn_neighbours: PositiveInt = 20
    knn_fit_rows: PositiveInt = 50_000
    knn_query_rows: PositiveInt = 10_000
    diagnostic_rows: PositiveInt = 16_384
    sigreg_directions: PositiveInt = 256
    neighbour_queries: PositiveInt = 2000
    neighbour_pool: PositiveInt = 20_000
    min_isoscore: float = 0.8
    """Success: IsoScore at least this on held-out videos (the target criterion of §4.4.3)."""
    max_r2_drop: float = 0.02
    """Success: linear R² of positions and velocities within this of the raw latent."""
    max_sigreg_ratio: float = 2.0
    """Success: SIGReg at most this times the value of a true Gaussian of the same size."""
    seed: int = 0


class VideoEncoderSettings(FrozenModel):
    family: Literal["vjepa2_1", "vjepa2_hf"]
    entrypoint: str | None = None
    """torch.hub entry point of a V-JEPA 2.1 model, e.g. ``vjepa2_1_vit_large_384``."""
    checkpoint: str | None = None
    """Name of the checkpoint in ``checkpoints`` (V-JEPA 2.1)."""
    model_id: str | None = None
    """Hugging Face repository (V-JEPA 2), read from the local cache."""


class VideoRun(FrozenModel):
    """One set of frozen video features: an encoder, a resolution and a choice of frames."""

    encoder: str
    size: PositiveInt = 256
    """Side of the square crop; other than the stored clips' side, it is re-cut from the source."""
    frames: Literal["selected", "contiguous"] = "selected"
    clips: PositiveInt | None = None
    """At most this many of the read-out clips (the first ones); all of them when unset."""


class VideoProbeSettings(FrozenModel):
    """PC2, PC3, PC4 and the frame-selection collaudo on frozen video features (§4.12.1)."""

    hub_repo: Path
    """Local copy of the torch.hub repository facebookresearch/vjepa2 (nodes are offline)."""
    encoders: dict[str, VideoEncoderSettings] = Field(default_factory=dict)
    runs: dict[str, VideoRun] = Field(default_factory=dict)
    batch_size: PositiveInt = 4
    keypoint_clips: PositiveInt = 4000
    """Clips of the hand read-outs; the text read-out uses every extracted clip."""
    encoder_runs: list[str] = Field(default_factory=list)
    """PC3: the runs compared to choose the encoder, same resolution and frames."""
    default_encoder_run: str = ""
    """PC3: the run chosen on a tie (§4.12.1: V-JEPA 2.1-L)."""
    low_resolution_run: str = ""
    high_resolution_run: str = ""
    min_hand_gain: float = 0.05
    """PC4: 384 only if the hand read-out gains more than this R²."""
    contiguous_run: str = ""
    frame_tolerance: float = 0.05
    """Collaudo: R² with the selected frames must not fall more than this below contiguous."""
    reference_clips: PositiveInt = 20
    """Clips of the weight-reproduction checks (§4.13.1: 20 clips, 20 pose sequences)."""
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
    video_probes: VideoProbeSettings | None = None
    pose_isotropy: PoseIsotropySettings = Field(default_factory=PoseIsotropySettings)
    test_data: TestDataSettings

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        with path.open(encoding="utf-8") as stream:
            return cls.model_validate(yaml.safe_load(stream))

    def acquisition(self) -> AcquisitionConfig:
        return AcquisitionConfig.from_yaml(self.acquisition_config)

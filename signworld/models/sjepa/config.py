"""The hyperparameters of S-JEPA's pre-training on NTU (paper §4.2, MAMP's configuration)."""

from typing import Self

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from signworld.data.acquisition.config import FrozenModel

NTU_LEFT_RIGHT: tuple[tuple[int, int], ...] = (
    (4, 8),
    (5, 9),
    (6, 10),
    (7, 11),
    (12, 16),
    (13, 17),
    (14, 18),
    (15, 19),
    (21, 23),
    (22, 24),
)
"""NTU-25 joints swapped by a flip: shoulders, elbows, wrists, hands, hips, knees, ankles,
feet, hand tips, thumbs (0-based)."""


class SkeletonViewSettings(FrozenModel):
    """The views of the view encoder (paper §3, «Views»)."""

    rotation: float = Field(default=0.3, ge=0.0)
    """Half-width, in radians, of the uniform rotation about the skeleton's vertical axis. The
    paper gives no range: MAMP's ``random_rot`` default [Aperto]."""
    translation: float = Field(default=0.1, ge=0.0)
    """Half-width of the uniform translation, in the input's units [Aperto: not in the paper]."""
    flip_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    """Probability of the spatial flip [Aperto: not in the paper]."""
    pelvis: int = 0
    """Joint at the bottom of the vertical axis (NTU: base of the spine)."""
    neck: int = 20
    """Joint at its top (NTU: spine between the shoulders)."""
    left_right: tuple[tuple[int, int], ...] = NTU_LEFT_RIGHT


class SJEPASettings(FrozenModel):
    """S-JEPA on NTU, as the paper sets it."""

    frames: PositiveInt = 120
    """T: every sequence is trimmed and resized to it."""
    joints: PositiveInt = 25
    channels: PositiveInt = 3
    segment: PositiveInt = 4
    """l: frames embedded together per joint."""
    width: PositiveInt = 256
    """C_e = C_p."""
    depth: PositiveInt = 8
    """L_e: blocks of the view and target encoders."""
    predictor_depth: PositiveInt = 5
    """L_p."""
    heads: PositiveInt = 8
    mlp_ratio: PositiveFloat = 4.0
    """Hidden width 1,024."""
    mask_ratio: float = Field(default=0.9, gt=0.0, lt=1.0)
    motion_temperature: PositiveFloat = 0.80
    """τ of MAMP's masking: 0.80 for NTU-60 and PKU-MMD, 0.75 for NTU-120 (MAMP's code)."""
    centre_rate: float = Field(default=0.9, ge=0.0, lt=1.0)
    """β of the centring."""
    student_temperature: PositiveFloat = 0.1
    """τ_p."""
    teacher_temperature: PositiveFloat = 0.06
    """τ_t."""
    momentum: tuple[float, float] = (0.9999, 1.0)
    """λ of the target encoder's EMA, from its first to its last value on a cosine."""
    epochs: PositiveInt = 1200
    warmup_epochs: PositiveInt = 20
    batch_size: PositiveInt = 256
    learning_rate: PositiveFloat = 1e-3
    """Peak, reached linearly over ``warmup_epochs``."""
    min_learning_rate: PositiveFloat = 5e-4
    """End of the cosine decay."""
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    trim: tuple[float, float] = (0.5, 1.0)
    """Share of the valid frames kept by the random trim before the resize, in training."""
    test_trim: float = 0.9
    """The share kept, centred, at test time."""
    views: SkeletonViewSettings = Field(default_factory=SkeletonViewSettings)

    @model_validator(mode="after")
    def _shapes(self) -> Self:
        if self.frames % self.segment:
            raise ValueError(f"{self.frames} frames do not split into segments of {self.segment}")
        if self.width % self.heads:
            raise ValueError(f"{self.heads} heads do not split a width of {self.width}")
        if self.warmup_epochs >= self.epochs:
            raise ValueError("the warm-up must end before the last epoch")
        return self

    @property
    def segments(self) -> int:
        return self.frames // self.segment

    @property
    def tokens(self) -> int:
        """N = T_e x V: one token per segment and joint."""
        return self.segments * self.joints

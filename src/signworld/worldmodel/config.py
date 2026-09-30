"""Configuration of a WorldSign run, from one or more YAML files (§4.8-§4.10, §4.14).

A run is described by ``configs/model/worldsign.yaml`` plus optional overlays applied in
order, each overriding only the keys it names: one per loss arm of ESP-1
(``configs/model/ablations/arm_*.yaml``) and one per ablation on the best arm
(``esp2_*``, ``esp3_*``, ``esp4_*``). A file may also name a parent with ``inherits``.
Every value that the project document leaves open is marked ``[Aperto]`` next to it.
"""

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from ..acquisition.config import FrozenModel

Arm = Literal["A0", "A", "B0", "B", "C"]
"""Loss arms of ESP-1 (§4.14): alignment with or without SIGReg and L_unif, or InfoNCE."""


class LoRASettings(FrozenModel):
    rank: PositiveInt = 16
    alpha: PositiveFloat = 16.0
    """Scale alpha of ``(alpha / r) B A``; §4.4.2 gives the formula, not the value [Aperto]."""
    targets: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2")
    """Linear layers adapted in every block; ``qkv`` gets one adapter each for q, k and v."""


class EncoderSettings(FrozenModel):
    """V-JEPA 2.1 ViT-L distilled, frozen, adapted with LoRA on every block (§4.4.1)."""

    hub_repo: Path
    entrypoint: str = "vjepa2_1_vit_large_384"
    checkpoint: Path
    levels: tuple[int, ...] = (5, 11, 17, 23)
    """Blocks 6/12/18/24 (0-based), as V-JEPA 2.1's own ``hierarchical_layers``."""
    lora: LoRASettings = Field(default_factory=LoRASettings)
    train_norms: bool = True
    """The encoder's LayerNorms train too: the 0.10 M «LayerNorm e bias» of §4.8."""
    activation_checkpointing: bool = True


class MaskSpec(FrozenModel):
    """One of V-JEPA's multi-block masks: ``blocks`` tubes of ``spatial_scale`` of the frame."""

    blocks: PositiveInt
    spatial_scale: float
    temporal_scale: float = 1.0
    """1.0: every block spans all the steps (tubes)."""
    aspect_ratio: tuple[float, float] = (0.75, 1.5)


class MaskingSettings(FrozenModel):
    """V-JEPA's masks unchanged (§4.5.2): both kinds are drawn at every step."""

    specs: tuple[MaskSpec, ...] = (
        MaskSpec(blocks=8, spatial_scale=0.15),
        MaskSpec(blocks=2, spatial_scale=0.7),
    )


class FusionSettings(FrozenModel):
    """Input of the physical predictor from the four levels (§4.4.5).

    ``mlp`` is V-JEPA 2.1's own (Linear 4 D → D, GELU, Linear D → P), standard init;
    ``linear`` is one Linear 4 D → P starting as the released layer on the last level;
    ``residual`` is the released layer plus the ``mlp`` branch with its last layer at zero.
    Only the last two reproduce the released predictor at step 0 (PC6) [Aperto].
    """

    kind: Literal["mlp", "linear", "residual"] = "mlp"
    hidden: PositiveInt = 1024


class PhysicalSettings(FrozenModel):
    enabled: bool = True
    """False for ESP-2: no pose encoder, physical predictor, anchor or pose SIGReg."""
    lora: LoRASettings = Field(default_factory=LoRASettings)
    target_dim: PositiveInt = 256
    """C, the width of the pose latent s_{t,a} (S-JEPA, PC5)."""
    mask_index: int = 0
    """Only the first mask token of the released predictor is trained (PC6)."""
    context_lambda: float = 0.5
    """λ of the visible-token term (V-JEPA 2.1)."""
    lambda_warmup: tuple[float, float] = (15_000 / 252_000, 30_000 / 252_000)
    """Warm-up of λ from 0 to its value, as fractions of the run: V-JEPA 2.1's 15k-30k of 252k."""
    box_threshold: float = 0.3
    """Keypoint score above which a joint shapes its articulator's box [Aperto: 0.3 or 1.0]."""


class SemanticSettings(FrozenModel):
    """Predictor of the caption embedding, VL-JEPA's scheme at 7.7 M parameters (§4.4.6)."""

    width: PositiveInt = 384
    depth: PositiveInt = 4
    heads: PositiveInt = 12
    queries: PositiveInt = 8
    hypotheses: PositiveInt = 1
    """K of the latent variable: 1 in every run, 4 in the optional ESP-4 (§4.5.9)."""
    relaxation: float = 0.05
    """ε: share of the gradient the losing hypotheses receive in ESP-4 [Aperto: PC7]."""
    output_dim: PositiveInt = 512
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    """[Aperto: PC7] §4.11 asks for dropout without a value."""
    drop_path: float = 0.1
    """[Aperto: PC7] stochastic depth, §4.11."""
    layer_scale: float = 1e-4
    """[Aperto: PC7] initial LayerScale of the residual branches, §4.11."""

    @model_validator(mode="after")
    def _groups(self) -> Self:
        if self.queries % self.hypotheses:
            raise ValueError("the queries must split evenly into the hypotheses")
        return self


class LossSettings(FrozenModel):
    """Objective of §4.5.7: ``(1 - λ)·(predictive terms) + λ·(SIGReg terms)``."""

    arm: Arm = "A"
    sigreg_weight: float = 0.05
    sigreg_directions: PositiveInt = 1024
    """Random directions per SIGReg evaluation, redrawn at every step (LeJEPA recommends 1,024)."""
    sigreg_knots: PositiveInt = 17
    uniformity_t: PositiveFloat = 2.0
    """t of Wang and Isola's L_unif [Lett. 40]."""
    infonce_temperature: PositiveFloat = 0.07
    """Initial, learnable temperature of arm C (§4.10)."""


class SamplingSettings(FrozenModel):
    """The 64 frames of a sentence (§3.6): half by local motion, half uniform in time."""

    strategy: Literal["motion", "uniform"] = "motion"
    frames: PositiveInt = 64
    uniform_fraction: float = 0.5
    stride: PositiveInt = 3
    block: PositiveInt = 8
    top_blocks: PositiveInt = 8


class AugmentationSettings(FrozenModel):
    """Views, not samples (§3.8): box jitter on video and keypoints alike, colour on video."""

    box_jitter: float = 0.10
    brightness: float = 0.2
    """[Aperto] §3.8 names colour and brightness without values."""
    contrast: float = 0.2
    saturation: float = 0.2


class PoseEncoderSettings(FrozenModel):
    """S-JEPA as the physical target; implemented with the pose branch (§4.4.3)."""

    trainable: bool = True
    """False for ESP-3: frozen, no LoRA, SIGReg on the pose only as a diagnostic."""
    lora_rank: PositiveInt = 4
    learning_rate_multiplier: float = 0.05


class TrainingSettings(FrozenModel):
    """Starting values of §4.10, to be calibrated in the dry run (PC7)."""

    batch_size: PositiveInt = 128
    epochs: PositiveInt = 6
    patience: PositiveInt = 2
    learning_rate: PositiveFloat = 2e-4
    """[Aperto: PC7] searched in {1e-4, 2e-4, 5e-4}."""
    betas: tuple[float, float] = (0.9, 0.999)
    """AdamW as V-JEPA 2.1."""
    weight_decay: float = 0.04
    """V-JEPA 2.1's value, held constant (its ``final_weight_decay`` equals it)."""
    gradient_clip: PositiveFloat | None = None
    """None: V-JEPA 2.1 does not clip gradients."""
    warmup_fraction: float = 0.05
    cooldown_fraction: float = 0.05
    stages: dict[str, float] = Field(default_factory=lambda: {"1a": 0.01, "1": 0.05, "2a": 0.01})
    """Curriculum stages as fractions of the steps [Aperto: PC7]."""
    precision: Literal["bf16", "fp32"] = "bf16"
    seed: int = 0


class WorldSignConfig(FrozenModel):
    name: str
    encoder: EncoderSettings
    masking: MaskingSettings = Field(default_factory=MaskingSettings)
    fusion: FusionSettings = Field(default_factory=FusionSettings)
    physical: PhysicalSettings = Field(default_factory=PhysicalSettings)
    semantic: SemanticSettings = Field(default_factory=SemanticSettings)
    losses: LossSettings = Field(default_factory=LossSettings)
    sampling: SamplingSettings = Field(default_factory=SamplingSettings)
    augmentation: AugmentationSettings = Field(default_factory=AugmentationSettings)
    pose_encoder: PoseEncoderSettings = Field(default_factory=PoseEncoderSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive merge: mappings merge key by key, anything else is replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream) or {}
    parent = raw.pop("inherits", None)
    if parent is None:
        return raw
    return merge(_read((path.parent / parent).resolve()), raw)


def load_config(*paths: Path) -> WorldSignConfig:
    """The configuration of ``paths[0]`` with every later file laid over it, in order."""
    if not paths:
        raise ValueError("at least one configuration file is needed")
    merged: dict[str, Any] = {}
    for path in paths:
        merged = merge(merged, _read(path))
    return WorldSignConfig.model_validate(merged)

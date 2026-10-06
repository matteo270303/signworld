"""Configuration of a WorldSign run, from one or more YAML files (§4.8-§4.10, §4.14).

A run is described by ``parameters/model/worldsign.yaml`` plus optional overlays applied in
order, each overriding only the keys it names: one per loss arm of ESP-1
(``parameters/ablation/arm_*.yaml``) and one per ablation on the best arm
(``esp2_*``, ``esp4_*``, ...). A file may also name a parent with ``inherits``.
Every value that the project document leaves open is marked ``[Aperto]`` next to it. The pose
encoder is described in ``docs/worldsign-posa.md``, the stages and learning rates in
``docs/worldsign-gerarchia.md``.
"""

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from signworld.data.acquisition.config import FrozenModel
from signworld.metrics.directions import Bidirectional

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
    checkpoint_url: str | None = None
    """Where the weights are fetched from when ``checkpoint`` is missing; None: Meta's release."""
    checkpoint_sha256: str | None = None
    """Expected SHA-256 of ``checkpoint`` (P5); None: recorded at the first launch."""
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
    target_dim: PositiveInt = 192
    """C, the width of the pose target s_t (``pose_encoder.output_dim``)."""
    mask_index: int = 0
    """Only the first mask token of the released predictor is trained (PC6)."""
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    """Dropout in the blocks of the physical predictor, after the attention projection and in
    the MLP: LeWorldModel's 0.1, which lifted its planning success from 78 to 96 %."""
    context_lambda: float = 0.5
    """λ of the visible-token term (V-JEPA 2.1)."""
    lambda_progressive: bool = False
    """False: λ from the first step, as V-JEPA 2.1's cooldown (``lambda_progressive: false``),
    which is our regime: 64-frame clips from an already trained model. True: its pre-training
    warm-up, over ``lambda_warmup``."""
    lambda_warmup: tuple[float, float] = (15_000 / 252_000, 30_000 / 252_000)
    """Warm-up of λ from 0 to its value, as fractions of the run: V-JEPA 2.1's 15k-30k of 252k."""
    weight_distance: bool = False
    """False: every visible token weighs 1 in L_ctx, as V-JEPA 2.1's cooldown
    (``weight_distance_loss: false``). True: ``1 / √d``, as its pre-training."""
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
    trains_encoder: bool = False
    """False: the hierarchy trained level by level, the semantic level reads the video encoder
    without changing it (``sg(Enc(x))``). True: the «global» ablation, E_sem reaches the LoRA."""

    @model_validator(mode="after")
    def _groups(self) -> Self:
        if self.queries % self.hypotheses:
            raise ValueError("the queries must split evenly into the hypotheses")
        return self


class LossSettings(FrozenModel):
    """Objective of §4.5.7: ``(1 - λ)·(predictive terms) + λ·(SIGReg terms)``, level by level."""

    arm: Arm = "A"
    sigreg_weight: float = 0.05
    """λ of the levels above the pose."""
    pose_sigreg_weight: float = Field(default=0.04, gt=0.0, lt=1.0)
    """λ of the pose level (posa §4.2): LeJEPA's 0.02 for four views at 256 samples per
    SIGReg, doubled for the ≤ 128 clips of each step, since the statistic grows with N."""
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

    enabled: bool = True
    """False: every clip is seen as it was cropped, at every epoch."""
    box_jitter: float = 0.10
    brightness: float = 0.2
    """[Aperto] §3.8 names colour and brightness without values."""
    contrast: float = 0.2
    saturation: float = 0.2


class PoseViewSettings(FrozenModel):
    """The views of the pose's invariance term besides the clean sequence (posa §4.1).

    ``count`` independent draws of one pipeline of nuisances, as LeJEPA's views: one linear
    map of the camera per clip, Gaussian noise on the present joints at the detector's jitter,
    and joints hidden over a span of steps.
    """

    count: PositiveInt = 3
    """Draws besides the clean sequence: four views in all, LeJEPA's minimal example."""
    rotation_degrees: float = Field(default=10.0, ge=0.0)
    """Half-width of the uniform in-plane rotation about the origin between the shoulders."""
    aspect: float = Field(default=0.15, ge=0.0, lt=1.0)
    """Half-width of the uniform horizontal scale around 1 (SPOTER's squeeze, up to 15 %)."""
    shear: float = Field(default=0.1, ge=0.0)
    """Half-width of the uniform shear ``x ← x + h·y`` [our choice]."""
    affine_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    """Chance of the aspect and, apart, of the shear; the rotation is always drawn."""
    noise_body: float = Field(default=0.003, ge=0.0)
    noise_hands: float = Field(default=0.01, ge=0.0)
    noise_face: float = Field(default=0.002, ge=0.0)
    """Standard deviations of the noise, in shoulder units: the jitter measured on OpenASL from
    three consecutive frames (an upper bound: body 0.003, hands 0.008-0.009, face 0.002)."""
    mask_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    """Chance that a draw hides joints."""
    mask_steps: PositiveInt = 8
    """Consecutive steps over which they are hidden."""
    mask_joints: int = Field(default=9, ge=0)
    """Joints hidden, among the fingers and the face without their roots (57): PSTL's 9."""


class PoseEncoderSettings(FrozenModel):
    """The pose encoder, trained from scratch with the rest of the model (posa §3-§4).

    worldSign's part-based encoder on the V-JEPA tubelet: one spatial transformer per
    articulator, their tokens concatenated, a temporal transformer at ``4 x part_width`` and a
    final LayerNorm and projection to ``output_dim``.
    """

    part_width: PositiveInt = 128
    part_depth: PositiveInt = 2
    """Blocks of each articulator's spatial transformer."""
    depth: PositiveInt = 2
    """Blocks of the temporal transformer, at the width of the four parts concatenated (512)."""
    heads: PositiveInt = 8
    mlp_ratio: PositiveFloat = 4.0
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    """0: the target sg(s) the physical level reads must not change with a dropout draw."""
    output_dim: PositiveInt = 192
    """C, the width of the target s_t, after the final projection: LeWM's best embedding size,
    above the 125 linear components of 99.9 % of a step's pose on OpenASL."""
    learning_rate: PositiveFloat = 3e-4
    """Peak learning rate of the pose family (worldSign's value)."""
    warmup_epochs: PositiveFloat = 2.0
    """Linear warm-up over these epochs, stage P, then a cosine to 0 at the planned end of the
    run: the target is already slowing down when the physical level starts chasing it."""
    views: PoseViewSettings = Field(default_factory=PoseViewSettings)

    @model_validator(mode="after")
    def _heads(self) -> Self:
        if self.part_width % self.heads:
            raise ValueError(f"{self.heads} heads do not split a part width of {self.part_width}")
        return self


class TextSettings(FrozenModel):
    """The text branch after the pre-computed EmbeddingGemma rows (§4.4.4)."""

    input_dim: PositiveInt = 768
    """EmbeddingGemma's whole vector: no truncation (29/9)."""
    hidden: PositiveInt = 512
    output_dim: PositiveInt = 512
    dropout: float = 0.1
    """Between the two layers of the head: «dropout nelle teste» of §4.11 [Aperto: PC7]."""


class StageSettings(FrozenModel):
    """Epochs of the stages before F, which lasts to the end (gerarchia §6.1)."""

    pose_epochs: PositiveInt = 2
    """Stage P: the pose and semantic levels alone, while the pose's learning rate warms up."""
    heads_epochs: PositiveInt = 1
    """Stage F0: plus the new modules of the physical level, every LoRA still frozen."""


class TrainingSettings(FrozenModel):
    """Starting values of §4.10, to be calibrated in the dry run (PC7)."""

    batch_size: PositiveInt = 128
    epochs: PositiveInt = 15
    patience: PositiveInt = 3
    """Epochs without a better decision metric before the cooldown, counted in stage F only."""
    learning_rate: PositiveFloat = 2e-4
    """[Aperto: PC7] searched in {1e-4, 2e-4, 5e-4}."""
    betas: tuple[float, float] = (0.9, 0.999)
    """AdamW as V-JEPA 2.1."""
    weight_decay: float = 0.04
    """V-JEPA 2.1's value, held constant (its ``final_weight_decay`` equals it)."""
    gradient_clip: PositiveFloat | None = None
    """None: V-JEPA 2.1 does not clip gradients."""
    activation_warmup_epochs: PositiveFloat = 2.0
    """Linear warm-up of every family but the pose from the step it enters training."""
    cooldown_fraction: float = Field(default=0.05, gt=0.0, lt=1.0)
    """V-JEPA 2's cooldown: the learning rate falls linearly to 0 over this share of the run."""
    stages: StageSettings = Field(default_factory=StageSettings)
    precision: Literal["bf16", "fp32"] = "bf16"
    seed: int = 0
    validation_every: PositiveInt = 4_000
    """Steps between validations and checkpoints: ~500,000 clips at batch 128 (§4.10). With the
    diagnostics at the end of each epoch (``diagnostics.cadence``), between checkpoints only."""
    log_every: PositiveInt = 20
    find_unused_parameters: bool = False
    """DDP's search for parameters a step did not use; the curriculum freezes them instead."""
    preflight_overfit_steps: PositiveInt = 30
    """P13: steps of the single-batch overfit, with SIGReg off, before a run starts."""

    @model_validator(mode="after")
    def _stages_fit(self) -> Self:
        before = self.stages.pose_epochs + self.stages.heads_epochs
        if before >= self.epochs:
            raise ValueError(
                f"stages P and F0 take {before} of {self.epochs} epochs: stage F never starts"
            )
        return self


class DiagnosticsSettings(FrozenModel):
    """The fail-fast system during training (§4.13.3-§4.13.5).

    Cadences are in steps of 128 clips: the document's 100 / 500 / 2,000 steps of 1,024 clips
    become 800 / 4,000 / 16,000. Thresholds are starting points to calibrate in PC7.
    """

    cadence: Literal["steps", "epoch"] = "steps"
    """When the readings run. ``steps``: the frequent readings, the validation and the rare
    readings every ``frequent_every``, ``validation_every`` and ``rare_every`` steps, and the
    programmed stops at their steps. ``epoch``: all of them at the end of every epoch only, the
    frequent readings on its last training batch and the stops due within the epoch judged on
    those readings. Either way everything is read at step 0, the reference, and the end of the
    constant phase is validated, since the cooldown starts from the best checkpoint."""
    frequent_every: PositiveInt = 800
    """Collapse, predictors, read-out, dynamics, keypoints, LoRA, queries, gradients."""
    rare_every: PositiveInt = 16_000
    """Temporal order, encoder drift and the plausibility tests (every 2,000 of the document's
    steps, §4.12.4)."""
    diagnostic_clips: PositiveInt = 32
    """Clips per GPU in the extra diagnostic pass of the frequent readings."""
    probe_clips: PositiveInt = 256
    """The fixed probe batch: pose target content and isotropy, CKA, encoder drift."""
    small_clips: PositiveInt = 8
    """Clips per GPU for the leak test and the attention of the queries."""
    order_clips: PositiveInt = 64
    """Validation clips per GPU for the temporal order (omega)."""
    plausibility_clips: PositiveInt = 32
    """Validation clips per GPU for the plausibility tests at the rare cadence."""
    plausibility_masks: PositiveInt = 8
    """Masks of each kind per clip for Ē_fis in the plausibility tests (as in the test)."""
    spike_sigma: PositiveFloat = 6.0
    collapse_ratio: float = 0.5
    gamma_min: float = 0.3
    leak_r2: float = 0.98
    leak_tolerance: float = 1e-4
    localization_min: float = 0.10
    order_max: float = 0.95
    drift_min: float = 0.5
    spearman_min: float = 0.8
    conflict_cosine: float = -0.3
    conflict_readings: PositiveInt = 5
    """Consecutive frequent readings (5 x 800 = 4,000 steps, the document's 500)."""
    lora_ratio_max: float = 0.1
    modality_gap_max: float = 0.95
    noise_drop_min: float = 0.5
    share_max: float = 0.9
    hubness_growth_max: float = 1.5
    sigreg_growth_max: float = 1.5
    pose_r2_drop_max: float = 0.02
    """Largest fall of the probe R² of the keypoints from s below its best so far."""
    rotation_max: float = 0.1
    """Largest gap between the channel correlations of s, from one probe reading to the next,
    after and before aligning it by a rotation: above it the target's basis turns while the
    physical read-out chases it (gerarchia §4.1) [our choice]."""
    pose_r2_min: float = 0.9
    """Stop F1: position R² of both hands from s on the probe batch [Aperto: PC7]."""
    isoscore_min: float = 0.8
    visible_r2_min: float = 0.9
    excluded_hands_max: float = 0.5
    query_cosine_max: float = 0.95
    duplicate_cosine: float = 0.95
    """Captions this close (EmbeddingGemma cosine) count as the same in the tolerant R@1."""
    chance_multiple: float = 5.0
    ridge_baseline: Bidirectional | None = None
    """R@1 of the ridge baseline of PC2 on the same split, as fractions, in both directions
    (``{t2v: …, v2t: …}``); stop F3 compares their mean with the metric that decides
    [Aperto]."""
    gate_stops: bool = False
    """True only for the gate run: a failed stop F1-F3 ends the run (§4.13.5)."""


class DataSettings(FrozenModel):
    """Where the clips come from and how they are split and loaded (§3.4, §4.10)."""

    index: Path | None = None
    """The training index of stage 0 (``build_training_index``), one row per usable clip."""
    embeddings: Path | None = None
    """The directory of the pre-computed EmbeddingGemma rows (``EmbeddingStore``)."""
    validation_fraction: float = 0.1
    """Share of the channels of each sign language held out for validation [Aperto]."""
    held_out_languages: tuple[str, ...] = ()
    """Sign languages kept out of training entirely: the held-out language split [Aperto]."""
    split_source: Literal["channel", "manifest"] = "channel"
    """``channel``: our splits by language, channel and video (§3.4). ``manifest``: the corpus's
    own splits (a benchmark's train / valid / test), for runs on a benchmark alone."""
    validation_split: str = "val_channel"
    """The split validated during training, whose R@1 decides early stopping."""
    seen_video_fraction: float = 0.02
    """Share of the videos of training channels held out, to measure the gap between seen
    channels and held-out channels (§4.13.4) [Aperto]."""
    manifest: Path | None = None
    """The corpus manifest, for the contamination assertion P2."""
    benchmarks: tuple[Path, ...] = ()
    """Manifests of the benchmarks (OpenASL, ...) whose validation and test clips must not
    overlap the corpus (§3.9, P2)."""
    validation_clips: PositiveInt = 2_000
    """Fixed subset of the held-out channel split validated during training (§4.10)."""
    statistics_clips: PositiveInt = 20_000
    """Training clips whose keypoints fix the scale of L_anchor."""
    workers: int = 8
    """Data loader processes per GPU."""
    prefetch: PositiveInt = 2


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
    text: TextSettings = Field(default_factory=TextSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    diagnostics: DiagnosticsSettings = Field(default_factory=DiagnosticsSettings)

    @model_validator(mode="after")
    def _widths(self) -> Self:
        if self.text.output_dim != self.semantic.output_dim:
            raise ValueError(
                f"the text head gives {self.text.output_dim} dimensions and the semantic "
                f"predictor {self.semantic.output_dim}: E_sem compares them"
            )
        if self.physical.enabled and self.physical.target_dim != self.pose_encoder.output_dim:
            raise ValueError(
                f"the physical head predicts {self.physical.target_dim} channels and the pose "
                f"encoder gives {self.pose_encoder.output_dim}"
            )
        return self

    @model_validator(mode="after")
    def _target_slows(self) -> Self:
        warmup, stage = self.pose_encoder.warmup_epochs, self.training.stages.pose_epochs
        if self.physical.enabled and warmup > stage:
            raise ValueError(
                f"the pose warms up over {warmup} epochs, past stage P ({stage}): the target "
                "would still be speeding up when the physical level starts chasing it"
            )
        return self


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

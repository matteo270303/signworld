"""The video side of WorldSign: one adapted encoder, two predictors (§4.2, gerarchia §2).

``VideoBranch.physical`` runs the physical pass of one training step: every mask kind of the
policy, the encoder on the visible tokens only, the multi-level fusion, the predictor and the
per-step read-out. ``VideoBranch.semantic`` runs the semantic pass on the whole clip. The
hierarchy is trained level by level: the semantic pass reads the encoder without gradient,
``sg(Enc(x))``, so only the physical level changes the LoRA; ``encoder_gradient`` lets it
through instead (the «global» ablation). ``build_video_branch`` assembles the branch from a
``WorldSignConfig``.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from signworld.data.pose.wholebody import Articulator
from signworld.experiment.train.config import WorldSignConfig
from signworld.models.encoders.video_encoders import load_vjepa2_1
from signworld.models.encoders.weights import VJEPA2_1_VITL_384_URL, ensure_encoder_files

from .backbone import VideoBackbone
from .fusion import build_fusion
from .masking import LambdaSchedule, Mask, MaskPolicy, TokenGrid, token_roles
from .physical import PhysicalPrediction, PhysicalPredictor
from .readout import StepReadout, membership
from .semantic import SemanticPredictor
from .video_target import TokenHeads, TokenPrediction, VideoTarget


@dataclass(frozen=True, slots=True)
class PhysicalOutput:
    predictions: list[PhysicalPrediction]
    """One per mask kind of the step, read per step towards the pose target; empty in ESP-6."""
    masks: list[Mask]
    context_lambda: float
    """λ of ``L_ctx`` at this step (warm-up of V-JEPA 2.1)."""
    tokens: list[TokenPrediction] = field(default_factory=list)
    """ESP-6: one per mask kind, every token towards the video target; empty with the pose."""


class VideoBranch(nn.Module):
    def __init__(  # noqa: PLR0913 (the parts of the branch and one loss option)
        self,
        backbone: VideoBackbone,
        physical: PhysicalPredictor | None,
        semantic: SemanticPredictor,
        masks: MaskPolicy,
        schedule: LambdaSchedule,
        grid: TokenGrid,
        *,
        weight_distance: bool = False,
        encoder_gradient: bool = False,
        target: VideoTarget | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.target = target
        """ESP-6: the frozen released encoder whose tokens the physical level predicts."""
        self.physical_predictor = physical
        self.semantic_predictor = semantic
        self.masks = masks
        self.schedule = schedule
        self.grid = grid
        self.weight_distance = weight_distance
        self.encoder_gradient = encoder_gradient
        """True: E_sem reaches the encoder (the «global» ablation); False: level by level."""

    @property
    def has_physical_level(self) -> bool:
        return self.physical_predictor is not None

    def physical(
        self,
        frames: Tensor,
        boxes: Tensor,
        visible: Tensor,
        step: int,
        total_steps: int,
        generator: torch.Generator,
    ) -> PhysicalOutput:
        """One physical pass per mask kind.

        ``frames`` (batch, 64, H, W, 3) uint8; ``boxes`` (batch, 32, 4, 4) in frame fractions;
        ``visible`` (batch, 32, 4).
        """
        if self.physical_predictor is None:
            raise RuntimeError("the physical level is disabled (ESP-2)")
        device = frames.device
        members = membership(boxes, visible, self.grid.rows, self.grid.columns)
        lam = self.schedule.at(step, total_steps)
        predictions, tokens, drawn = [], [], []
        for drawn_mask in self.masks(len(frames), generator):
            mask = drawn_mask.to(device)
            levels = self.backbone.context_levels(frames, mask.context)
            roles = token_roles(mask, self.grid, self.weight_distance)
            if self.target is None:
                predictions.append(self.physical_predictor(levels, mask, members, roles))
            else:
                tokens.append(self.physical_predictor.token_forward(levels, mask, roles))
            drawn.append(mask)
        return PhysicalOutput(predictions, drawn, lam, tokens)

    def target_tokens(self, frames: Tensor) -> Tensor:
        """ESP-6: (batch, N, width) the frozen encoder's tokens of the whole clip, normalised."""
        if self.target is None:
            raise RuntimeError("the physical level predicts the pose, not the video")
        tokens: Tensor = self.target(frames)
        return tokens

    def semantic(self, frames: Tensor, record: dict[str, Tensor] | None = None) -> Tensor:
        """(batch, K, d) predicted caption embeddings ŷ from the whole clip.

        The encoder runs without gradient unless ``encoder_gradient``: its tokens are then
        ``sg(Enc(x))`` and the pass keeps none of its activations. ``record``, if given,
        receives the encoder's mean token and every query's output, for the collapse and query
        diagnostics (§4.13.3).
        """
        with torch.set_grad_enabled(self.encoder_gradient and torch.is_grad_enabled()):
            tokens = self.backbone.tokens(frames)
        queries = self.semantic_predictor.query_outputs(tokens)
        if record is not None:
            record["encoder_mean"] = tokens.detach().mean(dim=1)
            record["queries"] = queries.detach()
        predicted: Tensor = self.semantic_predictor.project(queries)
        return predicted


def build_video_branch(config: WorldSignConfig, grid: TokenGrid | None = None) -> VideoBranch:
    """Load the released V-JEPA 2.1 encoder and predictor offline and adapt them."""
    grid = grid or TokenGrid()
    encoder_settings = config.encoder
    ensure_encoder_files(
        encoder_settings.hub_repo,
        encoder_settings.checkpoint,
        url=encoder_settings.checkpoint_url or VJEPA2_1_VITL_384_URL,
        sha256=encoder_settings.checkpoint_sha256,
    )
    encoder, predictor, reports = load_vjepa2_1(
        encoder_settings.hub_repo, encoder_settings.entrypoint, encoder_settings.checkpoint
    )
    missing = [key for report in reports.values() for key in report.missing]
    if missing:
        raise RuntimeError(
            f"{len(missing)} weights missing from the checkpoint, e.g. {missing[:3]}"
        )
    return assemble(encoder, predictor, config, grid)


def assemble(
    encoder: nn.Module, predictor: Any, config: WorldSignConfig, grid: TokenGrid
) -> VideoBranch:
    """The branch around already-built Meta modules (also used by the tests, with tiny ones)."""
    # The physical level adapts the encoder; with neither it nor the global ablation, no level
    # trains it and it stays as released (ESP-2).
    adapted = config.physical.enabled or config.semantic.trains_encoder
    video_target = config.physical.enabled and config.physical.target == "video"
    # ESP-6's teacher is copied before the LoRA is injected in place: every weight as released.
    teacher = VideoTarget(encoder) if video_target else None
    backbone = VideoBackbone(encoder, config.encoder, adapted=adapted)
    physical = None
    if config.physical.enabled:
        released = predictor.predictor_embed
        fusion = build_fusion(config.fusion, backbone.level_norms(), released)
        width = int(predictor.predictor_norm.normalized_shape[0])
        readout: StepReadout | TokenHeads = (
            TokenHeads(width, teacher.width)
            if teacher is not None
            else StepReadout(width, len(Articulator), config.physical.target_dim)
        )
        physical = PhysicalPredictor(
            predictor,
            fusion,
            readout,
            config.physical.lora,
            grid,
            mask_index=config.physical.mask_index,
            activation_checkpointing=config.encoder.activation_checkpointing,
            dropout=config.physical.dropout,
        )
    semantic = SemanticPredictor(backbone.width, config.semantic, grid)
    settings = config.physical
    schedule = (
        LambdaSchedule(settings.context_lambda, *settings.lambda_warmup)
        if settings.lambda_progressive
        else LambdaSchedule.constant(settings.context_lambda)
    )
    return VideoBranch(
        backbone,
        physical,
        semantic,
        MaskPolicy(config.masking.specs, grid),
        schedule,
        grid,
        weight_distance=settings.weight_distance,
        encoder_gradient=config.semantic.trains_encoder,
        target=teacher,
    )

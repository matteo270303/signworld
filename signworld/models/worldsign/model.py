"""WorldSign: the video, pose and text branches with the objective of one arm (§4.2, §4.5).

``WorldSign.loss`` runs the physical pass, the semantic pass or both, as the curriculum of
§4.10 asks, and combines their terms as §4.5.7:
``(1 - λ)·(E_fis + L_anchor + L_pred_sem) + λ·(SIGReg_posa + SIGReg_sem)``. Nothing is
stopped: the pose target ``s`` receives the gradient of every term that reads it (§4.5.1).
In ESP-3 the pose encoder is frozen, so ``SIGReg_posa`` is only logged and the anchor trains
only its decoders; in ESP-2 there is no physical pass.

``parameter_groups`` gives AdamW its groups: the pose encoder at the learning rate x0.05, and
no weight decay on norms, biases, scalars and learned queries (§4.10).
"""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

from signworld.experiment.train.config import TrainingSettings, WorldSignConfig
from signworld.experiment.train.distributed import SINGLE, Distributed
from signworld.loss.worldsign import LossTerms, Objective, physical_energy, physical_energy_per_clip

from .masking import TokenGrid
from .pose_branch import PoseBranch, articulator_confidence
from .text_branch import TextBranch
from .video_branch import VideoBranch, build_video_branch


@dataclass(frozen=True, slots=True)
class WorldSignBatch:
    """One batch of clips, as the data pipeline delivers it."""

    frames: Tensor
    """(batch, 64, H, W, 3) uint8: the selected frames."""
    pose_tokens: Tensor
    """(batch, 32, 69, 6): S-JEPA's input, ``PoseSequence.tokens``."""
    keypoints: Tensor
    """(batch, 32, 69, 2) p̂: each joint's mean position over the two frames of a step."""
    keypoint_weights: Tensor
    """(batch, 32, 69) c_{t,j}: weight of each joint in L_anchor; averaged, c̄ of E_fis."""
    boxes: Tensor
    """(batch, 32, 4, 4) articulator box of each step, in frame fractions."""
    box_visible: Tensor
    """(batch, 32, 4) bool."""
    captions: Tensor
    """(batch, 768) pre-computed EmbeddingGemma rows."""
    languages: Tensor
    """(batch,) long: each caption's language, as ``LanguageCentering.index`` numbers it."""
    videos: Sequence[str]
    """Source video of each clip: InfoNCE does not use clips of one video as negatives."""
    caption_rows: Tensor | None = None
    """(batch,) long: each clip's caption row; validation counts equal captions as matches."""
    views: Tensor | None = None
    """(batch, 6) augmentation still to apply to ``frames``; None once applied or if none."""
    clip_ids: Sequence[str] | None = None
    durations: Tensor | None = None
    """(batch,) seconds of each clip, for the R@1 by duration band."""
    caption_words: Tensor | None = None
    """(batch,) words of each caption, for the R@1 by caption length."""

    def augmented(self) -> "WorldSignBatch":
        """The frames moved by their views (on the frames' device), which are then spent."""
        if self.views is None:
            return self
        # augmentation imports config only
        from signworld.data.augmentation import apply_views  # noqa: PLC0415

        return dataclasses.replace(self, frames=apply_views(self.frames, self.views), views=None)

    def take(self, count: int) -> "WorldSignBatch":
        """The first ``count`` clips."""
        taken: dict[str, Any] = {
            f.name: getattr(self, f.name)[:count]
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not None
        }
        taken["videos"] = list(self.videos[:count])
        if self.clip_ids is not None:
            taken["clip_ids"] = list(self.clip_ids[:count])
        return dataclasses.replace(self, **taken)

    def to(self, device: torch.device) -> "WorldSignBatch":
        moved = {
            f.name: getattr(self, f.name).to(device, non_blocking=True)
            for f in dataclasses.fields(self)
            if isinstance(getattr(self, f.name), Tensor)
        }
        return dataclasses.replace(self, **moved)


@dataclass(frozen=True, slots=True)
class StepRandomness:
    """The random draws of one step, reproducible from the seed, the step and the rank."""

    masks: torch.Generator
    """This GPU's own: every GPU draws different masks, as in V-JEPA."""
    directions: torch.Generator
    """The same on every GPU: SIGReg's slices must agree for its statistic to be one."""

    @classmethod
    def at(cls, seed: int, step: int, rank: int = 0) -> "StepRandomness":
        def generator(*key: int) -> torch.Generator:
            state = int(np.random.SeedSequence([seed, *key]).generate_state(1)[0])
            return torch.Generator().manual_seed(state)

        return cls(masks=generator(step, rank, 0), directions=generator(step, 1))


class WorldSign(nn.Module):
    def __init__(
        self,
        video: VideoBranch,
        pose: PoseBranch | None,
        text: TextBranch,
        objective: Objective,
    ) -> None:
        super().__init__()
        if video.has_physical_level != (pose is not None):
            raise ValueError("the physical predictor and the pose branch go together (ESP-2)")
        self.video = video
        self.pose = pose
        self.text = text
        self.objective = objective

    @property
    def has_physical_level(self) -> bool:
        return self.pose is not None

    def physical_terms(
        self,
        batch: WorldSignBatch,
        step: int,
        total_steps: int,
        randomness: StepRandomness,
        record: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """E_fis, L_anchor and SIGReg_posa (a diagnostic when the pose is frozen)."""
        pose = self.pose
        if pose is None:
            raise RuntimeError("the physical level is disabled (ESP-2)")
        latent = pose.targets(batch.pose_tokens)
        grid: TokenGrid = self.video.grid
        if latent.shape[1] != grid.steps:
            raise ValueError(f"{latent.shape[1]} pose steps for {grid.steps} video steps")
        confidence = articulator_confidence(batch.keypoint_weights)
        output = self.video.physical(
            batch.frames, batch.boxes, batch.box_visible, step, total_steps, randomness.masks
        )
        terms = {
            "e_fis": physical_energy(output.predictions, latent, confidence, output.context_lambda),
            "anchor": pose.anchor(latent, batch.keypoints, batch.keypoint_weights),
        }
        if record is not None:
            record |= {"latent": latent, "confidence": confidence, "physical": output}
        present = confidence > 0
        if pose.trainable:
            terms["sigreg_posa"] = self.objective.pose_sigreg(
                latent, present, randomness.directions
            )
            return terms, {}
        with torch.no_grad():
            diagnostic = self.objective.pose_sigreg(latent, present, randomness.directions)
        return terms, {"sigreg_posa": diagnostic}

    def semantic_terms(
        self,
        batch: WorldSignBatch,
        randomness: StepRandomness,
        record: dict[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        """The arm's L_pred_sem and SIGReg_sem, from ŷ of the whole clip and ẽ."""
        predicted = self.video.semantic(batch.frames, record)
        target = self.text(batch.captions, batch.languages)
        if record is not None:
            record |= {"predicted": predicted, "target": target}
        return self.objective.semantic_terms(predicted, target, batch.videos, randomness.directions)

    def loss(  # noqa: PLR0913 (the step, its randomness, the passes and a record)
        self,
        batch: WorldSignBatch,
        step: int,
        total_steps: int,
        randomness: StepRandomness,
        *,
        physical: bool = True,
        semantic: bool = True,
        record: dict[str, Any] | None = None,
    ) -> LossTerms:
        """The objective of one step; ``physical`` is ignored when the level is off (ESP-2).

        ``record``, if given, receives the intermediate tensors the diagnostics read: the pose
        target, the confidences, the physical read-outs, ŷ, ẽ and the query outputs.
        """
        terms: dict[str, Tensor] = {}
        diagnostics: dict[str, Tensor] = {}
        seen: dict[str, Any] = record if record is not None else {}
        if physical and self.has_physical_level:
            found, diagnostics = self.physical_terms(batch, step, total_steps, randomness, seen)
            terms.update(found)
        if semantic:
            terms.update(self.semantic_terms(batch, randomness, seen))
        if not terms:
            raise ValueError("no pass selected")
        combined = self.objective.combine(terms)
        return LossTerms(combined.total, combined.parts, diagnostics, self._per_clip(seen))

    @staticmethod
    @torch.no_grad()
    def _per_clip(seen: dict[str, Any]) -> dict[str, Tensor]:
        """Each clip's energies: E_sem through its best hypothesis, E_fis over the step's masks."""
        out: dict[str, Tensor] = {}
        if "predicted" in seen:
            cosine = functional.cosine_similarity(
                seen["predicted"], seen["target"][:, None], dim=-1
            )
            out["e_sem"] = (1 - cosine.amax(dim=1)).float()
        if "physical" in seen:
            output = seen["physical"]
            out["e_fis"] = physical_energy_per_clip(
                output.predictions, seen["latent"], seen["confidence"], output.context_lambda
            ).float()
        return out

    forward = loss
    """DDP synchronises the gradients of what ``forward`` computes: the whole step."""

    def parameter_groups(self, training: TrainingSettings) -> list[dict[str, Any]]:
        """AdamW groups over the trainable parameters (§4.10).

        ``base``: everything at the base rate, on the main schedule. ``pose``: S-JEPA's LoRA at
        x0.05. ``pose_final``: the pose encoder's final layer, at its own peak and on its own
        schedule (``schedule`` names it for the trainer). No weight decay on norms, biases,
        scalars and learned queries.
        """
        kind: dict[int, str] = {}
        multipliers = {"base": 1.0}
        if self.pose is not None:
            settings = self.pose.encoder.settings
            kind |= {id(p): "pose" for p in self.pose.encoder.parameters()}
            multipliers["pose"] = settings.learning_rate_multiplier
            if self.pose.encoder.final is not None:
                kind |= {id(p): "pose_final" for p in self.pose.encoder.final.parameters()}
                multipliers["pose_final"] = settings.final_layer_learning_rate_multiplier
        groups: dict[tuple[str, bool], list[nn.Parameter]] = {}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            decay = parameter.ndim > 1 and not name.endswith(("bias", "queries"))
            groups.setdefault((kind.get(id(parameter), "base"), decay), []).append(parameter)
        return [
            {
                "name": f"{group}-{'decay' if decay else 'no_decay'}",
                "params": parameters,
                "lr": training.learning_rate * multipliers[group],
                "weight_decay": training.weight_decay if decay else 0.0,
                "schedule": "final_layer" if group == "pose_final" else "main",
            }
            for (group, decay), parameters in sorted(groups.items())
        ]


def assemble_worldsign(
    video: VideoBranch,
    pose: PoseBranch | None,
    config: WorldSignConfig,
    collective: Distributed = SINGLE,
) -> WorldSign:
    """The model around already-built branches (also used by the tests, with tiny ones)."""
    objective = Objective(
        config.losses, config.semantic, physical=config.physical.enabled, collective=collective
    )
    return WorldSign(video, pose, TextBranch(config.text), objective)


def build_worldsign(
    config: WorldSignConfig, grid: TokenGrid | None = None, collective: Distributed = SINGLE
) -> WorldSign:
    """V-JEPA 2.1 and S-JEPA loaded from their checkpoints, adapted as ``config`` says.

    The language means of the text branch and the keypoint variance of the anchor are fitted
    afterwards, on the training clips (``text.centering.fit``, ``pose.fit``).
    """
    video = build_video_branch(config, grid)
    pose = PoseBranch.from_settings(config.pose_encoder) if config.physical.enabled else None
    return assemble_worldsign(video, pose, config, collective)

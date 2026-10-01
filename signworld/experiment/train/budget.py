"""Trainable parameters per component, against the budget of §4.8 (22 M, assertion P3)."""

from dataclasses import dataclass

from torch import nn

from signworld.models.worldsign.lora import adapter_parameters
from signworld.models.worldsign.model import WorldSign
from signworld.models.worldsign.video_branch import VideoBranch


def _count(parameters: list[nn.Parameter]) -> int:
    return sum(p.numel() for p in parameters if p.requires_grad)


@dataclass(frozen=True, slots=True)
class VideoBudget:
    encoder_lora: int
    encoder_norms: int
    predictor_lora: int
    fusion: int
    physical_head: int
    semantic_predictor: int

    @property
    def total(self) -> int:
        return (
            self.encoder_lora
            + self.encoder_norms
            + self.predictor_lora
            + self.fusion
            + self.physical_head
            + self.semantic_predictor
        )

    @classmethod
    def of(cls, branch: VideoBranch) -> "VideoBudget":
        encoder = branch.backbone.encoder
        adapters = {id(p) for p in adapter_parameters(encoder)}
        norms = [
            p
            for m in encoder.modules()
            if isinstance(m, nn.LayerNorm)
            for p in m.parameters()
            if id(p) not in adapters
        ]
        physical = branch.physical_predictor
        return cls(
            encoder_lora=_count(adapter_parameters(encoder)),
            encoder_norms=_count(norms),
            predictor_lora=_count(adapter_parameters(physical.predictor)) if physical else 0,
            fusion=_count(list(physical.fusion.parameters())) if physical else 0,
            physical_head=_count(list(physical.readout.parameters())) if physical else 0,
            semantic_predictor=_count(list(branch.semantic_predictor.parameters())),
        )


@dataclass(frozen=True, slots=True)
class ModelBudget:
    """The whole model: the video branch plus the pose and text branches and the objective."""

    video: VideoBudget
    pose_lora: int
    pose_final_layer: int
    keypoint_decoders: int
    text_head: int
    objective: int
    """InfoNCE's temperature in arm C, else 0."""

    @property
    def total(self) -> int:
        return (
            self.video.total
            + self.pose_lora
            + self.pose_final_layer
            + self.keypoint_decoders
            + self.text_head
            + self.objective
        )

    @classmethod
    def of(cls, model: WorldSign) -> "ModelBudget":
        pose = model.pose
        final = pose.encoder.final if pose is not None else None
        return cls(
            video=VideoBudget.of(model.video),
            pose_lora=_count(adapter_parameters(pose.encoder)) if pose else 0,
            pose_final_layer=_count(list(final.parameters())) if final is not None else 0,
            keypoint_decoders=_count(list(pose.decoders.parameters())) if pose else 0,
            text_head=_count(list(model.text.parameters())),
            objective=_count(list(model.objective.parameters())),
        )

"""Trainable parameters per component, against the budget of §4.8 (22 M, assertion P3)."""

from dataclasses import dataclass

from torch import nn

from .lora import adapter_parameters
from .video_branch import VideoBranch


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

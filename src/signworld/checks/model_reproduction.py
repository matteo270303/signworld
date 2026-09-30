"""Collaudo "riproduzione dei pesi" (§4.13.1) and PC6 (§4.12.1).

* **Video encoder.** Every checkpoint key must load. Our input path (uint8 frames, ImageNet
  normalisation, channel-first layout) must give the same tokens as the official layout built
  independently, frame by frame, in float64: cosine per token > 0.999 and largest difference
  < 1e-3 in fp32. A deliberately wrong input (RGB swapped to BGR) must fail the same test, so
  the check is shown able to catch the error it exists for. bf16, which extraction uses, is
  compared with fp32 and reported.
* **S-JEPA.** The saved teacher is reloaded with every key and its latents on 20 fixed pose
  sequences are compared with reference latents. The pre-training run did not save any, so the
  first run writes them and says so; every later run (and the training code) compares.
* **PC6.** What the V-JEPA 2.1 predictor holds (input and output sizes, mask tokens, input
  projection, per-level norms of the encoder), and whether the multi-level input with the new
  levels at zero reproduces the released predictor exactly.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor, nn

from ..models.predictor_fusion import fused_embedding
from ..models.video_encoders import IMAGENET_MEAN, IMAGENET_STD, FrozenVideoEncoder, LoadReport

MIN_TOKEN_COSINE: Final = 0.999
MAX_ABS_DIFFERENCE: Final = 1e-3
FUSION_TOLERANCE: Final = 1e-4
"""Relative to the output scale: the fused predictor must compute the released one."""


@dataclass(frozen=True, slots=True)
class Agreement:
    min_cosine: float
    mean_cosine: float
    max_abs_difference: float

    @classmethod
    def of(cls, ours: Tensor, reference: Tensor) -> "Agreement":
        a, b = ours.flatten(0, -2).double(), reference.flatten(0, -2).double()
        cosine = torch.nn.functional.cosine_similarity(a, b, dim=1)
        return cls(float(cosine.min()), float(cosine.mean()), float((a - b).abs().max()))

    @property
    def reproduces(self) -> bool:
        return self.min_cosine > MIN_TOKEN_COSINE and self.max_abs_difference < MAX_ABS_DIFFERENCE


def official_input(frames: np.ndarray) -> Tensor:
    """(T, H, W, 3) uint8 to (1, 3, T, H, W) as V-JEPA's evaluation transforms lay it out,
    written independently of ``model_input``: per channel, float64, then cast."""
    video = frames.astype(np.float64) / 255.0
    for channel in range(3):
        video[..., channel] = (video[..., channel] - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
    return torch.from_numpy(video.transpose(3, 0, 1, 2)[None]).float()


@dataclass(frozen=True, slots=True)
class VideoReproduction:
    encoder: str
    clips: int
    load: dict[str, LoadReport]
    fp32: Agreement
    """Our path against the independently built official input."""
    swapped_channels: Agreement
    """Negative control: must NOT reproduce."""
    bf16: Agreement
    """bf16 extraction against fp32; reported, since bf16 cannot reach 1e-3."""
    deterministic: bool
    passed: bool


@torch.no_grad()
def video_reproduction(
    encoder: FrozenVideoEncoder, clips: list[np.ndarray], load: dict[str, LoadReport]
) -> VideoReproduction:
    ours, reference, swapped, low = [], [], [], []
    again = True
    for frames in clips:
        batch = torch.from_numpy(frames[None])
        tokens = encoder.tokens(batch, bf16=False)
        again &= bool(torch.equal(tokens, encoder.tokens(batch, bf16=False)))
        official = encoder.forward_video(official_input(frames).to(encoder.device)).float()
        ours.append(tokens.flatten(1, 3))
        reference.append(official)
        swapped.append(encoder.tokens(batch.flip(-1), bf16=False).flatten(1, 3))
        low.append(encoder.tokens(batch, bf16=True).flatten(1, 3))
    ours_all, reference_all = torch.cat(ours), torch.cat(reference)
    fp32 = Agreement.of(ours_all, reference_all)
    control = Agreement.of(torch.cat(swapped), reference_all)
    loaded = not any(report.missing for report in load.values())
    return VideoReproduction(
        encoder=encoder.name,
        clips=len(clips),
        load=load,
        fp32=fp32,
        swapped_channels=control,
        bf16=Agreement.of(torch.cat(low), ours_all),
        deterministic=again,
        passed=loaded and fp32.reproduces and not control.reproduces and again,
    )


@dataclass(frozen=True, slots=True)
class PoseReproduction:
    sequences: int
    reference: str
    created: bool
    """True when no reference existed: this run wrote it, nothing was compared."""
    agreement: Agreement | None
    deterministic: bool
    passed: bool


@torch.no_grad()
def pose_reproduction(encode: Any, tokens: Tensor, reference: Path) -> PoseReproduction:
    """``encode`` maps (n, 32, 69, 6) tokens to latents; compared with ``reference``."""
    latents = encode(tokens).float().cpu()
    deterministic = bool(torch.equal(latents, encode(tokens).float().cpu()))
    if not reference.is_file():
        reference.parent.mkdir(parents=True, exist_ok=True)
        np.savez(reference, tokens=tokens.numpy(), latents=latents.numpy())
        return PoseReproduction(
            len(tokens), str(reference), True, None, deterministic, deterministic
        )
    with np.load(reference) as saved:
        if not np.array_equal(saved["tokens"], tokens.numpy()):
            raise ValueError(f"{reference}: stored for other input sequences")
        agreement = Agreement.of(latents, torch.from_numpy(saved["latents"]))
    return PoseReproduction(
        len(tokens),
        str(reference),
        False,
        agreement,
        deterministic,
        deterministic and agreement.reproduces,
    )


@dataclass(frozen=True, slots=True)
class PredictorReport:
    """PC6 for one V-JEPA 2.1 checkpoint."""

    checkpoint: str
    load: dict[str, LoadReport]
    encoder_dim: int
    encoder_depth: int
    hierarchical_layers: list[int]
    """0-based blocks whose normalised outputs V-JEPA 2.1 treats as levels."""
    per_level_norms: int
    level_norm_drift: list[float]
    """Per level norm, largest distance of its weight from 1 and bias from 0: a norm still at
    its initial values was never trained (the distilled models supervise the last level only)."""
    predictor_input: str
    predictor_width: int
    predictor_depth: int
    predictor_output: int
    """Width of ``predictor_proj``: the teacher's (ViT-G) for the distilled models."""
    context_projection: bool
    mask_tokens: int
    trained_mask_tokens: list[int]
    """Mask tokens with non-zero weights: only these carry a learned "this is masked" query."""
    last_level_is_output: float
    """Largest difference between the last level and the encoder's default output."""
    fusion_difference: float
    """Largest difference, relative to the output scale, between the released predictor and
    the fused one with the new levels at zero."""
    passed: bool


def _gather(tokens: Tensor, indices: Tensor) -> Tensor:
    return torch.gather(tokens, 1, indices[..., None].expand(-1, -1, tokens.shape[-1]))


def _prediction(output: Any) -> Tensor:
    return output[0] if isinstance(output, tuple | list) else output  # type: ignore[no-any-return]


@torch.no_grad()
def predictor_report(
    name: str,
    encoder: FrozenVideoEncoder,
    predictor: Any,
    load: dict[str, LoadReport],
    frames: np.ndarray,
    seed: int = 0,
) -> PredictorReport:
    """``frames`` (T, H, W, 3) uint8 of one real clip, run in fp32."""
    module: Any = encoder.module
    layers = list(module.hierarchical_layers)
    batch = torch.from_numpy(frames[None])
    levels = encoder.levels(batch, layers)
    default = encoder.tokens(batch, bf16=False).flatten(1, 3)
    tokens = levels[-1].shape[1]
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(tokens, generator=generator)
    context = order[: tokens // 4].sort().values[None].to(encoder.device)
    target = order[tokens // 4 : tokens // 2].sort().values[None].to(encoder.device)

    # mask_index=0: only the first mask token of the released ViT-L is trained (worldSign,
    # predict_masked_hands); the default, 1, would seed every target with zeros.
    released = _prediction(
        predictor(_gather(levels[-1], context), [context], [target], mask_index=0)
    )
    embed = predictor.predictor_embed
    fusion = float("nan")
    if isinstance(embed, nn.Linear):
        predictor.predictor_embed = fused_embedding(embed, len(levels))
        try:
            stacked = torch.cat([_gather(level, context) for level in levels], dim=-1)
            fused = _prediction(predictor(stacked, [context], [target], mask_index=0))
        finally:
            predictor.predictor_embed = embed
        fusion = float((fused - released).abs().max() / released.abs().max().clamp_min(1e-12))
    mask_tokens = list(predictor.mask_tokens)
    trained = [i for i, token in enumerate(mask_tokens) if float(token.norm()) > 0]
    last_vs_default = float((levels[-1] - default).abs().max())
    return PredictorReport(
        checkpoint=name,
        load=load,
        encoder_dim=int(module.embed_dim),
        encoder_depth=len(module.blocks),
        hierarchical_layers=layers,
        per_level_norms=len(module.norms_block),
        level_norm_drift=[
            float(max((norm.weight - 1).abs().max(), norm.bias.abs().max()))
            for norm in module.norms_block
        ],
        predictor_input=repr(embed),
        predictor_width=int(predictor.predictor_norm.normalized_shape[0]),
        predictor_depth=len(predictor.predictor_blocks),
        predictor_output=int(predictor.predictor_proj.out_features),
        context_projection=hasattr(predictor, "predictor_proj_context"),
        mask_tokens=len(mask_tokens),
        trained_mask_tokens=trained,
        last_level_is_output=last_vs_default,
        fusion_difference=fusion,
        passed=fusion < FUSION_TOLERANCE and last_vs_default < MAX_ABS_DIFFERENCE,
    )

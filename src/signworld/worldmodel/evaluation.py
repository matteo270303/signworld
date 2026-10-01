"""The test of a trained model (§4.12.2-§4.12.4).

* **Every validation measure** on the test clips (``measures``): retrieval both ways with a
  bootstrap interval for every R@k, Precision@k, Recall@k, MRR, MedR, the tolerant R@1, R@1
  by language, clip duration and caption length; the losses; E_sem by language; the physical
  read-outs and keypoint errors; alignment and uniformity; the geometry of ŷ and ẽ; the noise
  test, hubness, the modality gap, the 2x2 energy table; the leak test and the attention of
  the queries; the isotropy and kinematic content of the pose target, as the probe batch
  reads them in validation.
* **Temporal order ω** (video reversed) on every clip.
* **Gate (F4)**: on OpenASL's test clips, T2V R@1 against the ridge baseline of PC2 and half of
  C²RL (``evaluation.gate``).
* **Plausibility** (violation of expectation, ``plausibility``) on every clip.
* **Hourglass (H5)**: accuracy of a linear probe of the sign language on the pose target, the
  video encoder's output and ŷ, on held-out videos.
"""

import dataclasses
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional

from ..evaluation.gate import GatePolicy
from ..metrics.probes import video_split
from .config import WorldSignConfig
from .measures import collect, model_measures, pose_target_measures, split_measures
from .model import WorldSign, WorldSignBatch
from .plausibility import PlausibilityResult, plausibility_tests
from .validation import EVERY_INTERVAL

# ------------------------------------------------------------------ probes


def language_probe(
    features: Tensor, labels: Tensor, videos: Sequence[str], steps: int = 300
) -> float:
    """Held-out accuracy of a linear probe of the sign language, split by video (H5)."""
    classes = int(labels.max()) + 1 if len(labels) else 0
    if classes < 2:  # noqa: PLR2004
        return float("nan")
    import numpy as np  # noqa: PLC0415

    test = torch.from_numpy(video_split(np.asarray(videos, dtype=object), 0.3))
    if bool(test.all()) or not bool(test.any()):
        return float("nan")
    x = features.float()
    mean, std = x[~test].mean(0), x[~test].std(0).clamp_min(1e-6)
    x = (x - mean) / std
    weight = torch.zeros(x.shape[1], classes, requires_grad=True)
    bias = torch.zeros(classes, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=steps)

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = (
            functional.cross_entropy(x[~test] @ weight + bias, labels[~test])
            + 1e-3 * weight.pow(2).sum()
        )
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    optimizer.step(closure)  # type: ignore[no-untyped-call]
    with torch.no_grad():
        predictions = (x[test] @ weight + bias).argmax(1)
    return float((predictions == labels[test]).float().mean())


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    measures: dict[str, float]
    gate: str | None
    plausibility: list[PlausibilityResult]
    language_probes: dict[str, float]

    def write(self, path: Path) -> None:
        payload = dataclasses.asdict(self)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


@torch.no_grad()
def _features(
    model: WorldSign, batches: Iterable[WorldSignBatch], device: torch.device
) -> dict[str, Tensor]:
    """Per clip: the pose target, the encoder's mean token and ŷ, for the language probes."""
    out: dict[str, list[Tensor]] = {"pose": [], "encoder": [], "semantic": []}
    for batch in batches:
        clips = batch.to(device)
        if model.pose is not None:
            out["pose"].append(
                model.pose.targets(clips.pose_tokens).float().mean(1).flatten(1).cpu()
            )
        tokens = model.video.backbone.tokens(clips.frames).float()
        out["encoder"].append(tokens.mean(1).cpu())
        out["semantic"].append(model.video.semantic_predictor(tokens).float().mean(1).cpu())
    return {k: torch.cat(v) for k, v in out.items() if v}


@torch.no_grad()
def order_cosine(
    model: WorldSign, batches: Iterable[WorldSignBatch], device: torch.device
) -> float:
    """ω = cos(ŷ(V), ŷ(V reversed)) over every clip: above ~0.95 the order is ignored."""
    was_training = model.training
    model.eval()
    cosines = []
    for batch in batches:
        frames = batch.frames.to(device)
        forward = model.video.semantic(frames).float().mean(1)
        backward = model.video.semantic(frames.flip(1)).float().mean(1)
        cosines.append(functional.cosine_similarity(forward, backward))
    model.train(was_training)
    return float(torch.cat(cosines).mean())


def evaluate(  # noqa: PLR0913 (the model, the clips, their labels and the options)
    model: WorldSign,
    batches: Iterable[WorldSignBatch],
    device: torch.device,
    config: WorldSignConfig,
    sign_languages: Tensor,
    videos: Sequence[str],
    *,
    masks: int = 8,
    ridge_baseline_r1: float | None = None,
    gate: bool = False,
) -> EvaluationReport:
    """Every measurement of §4.12.3-§4.12.4 on one set of clips, on one GPU.

    ``batches`` is read several times: a list, or a data loader that decodes again.
    """
    languages = model.text.centering.languages
    physical = model.pose is not None
    seen = collect(
        model, batches, device, bf16=device.type == "cuda", pose_target=physical, desc="[test]"
    )
    scores, measures = split_measures(seen, languages, config, bootstrap=EVERY_INTERVAL)
    first = next(iter(batches), None)
    if first is not None:
        measures |= model_measures(model, first, device, config.diagnostics.small_clips)
    if physical:
        t = seen.tensors
        target = pose_target_measures(t["latents"], t["keypoints"], t["weights"])
        measures |= {
            f"pose_{kind}_{name}": value
            for kind, values in target.items()
            for name, value in values.items()
        }
    measures["order_cosine"] = order_cosine(model, batches, device)
    decision = None
    if gate:
        if ridge_baseline_r1 is None:
            raise ValueError("the gate needs the ridge baseline of PC2")
        decision = GatePolicy().final(100 * scores.t2v[1], 100 * ridge_baseline_r1).value
    tests = plausibility_tests(model, batches, device, masks) if physical else []
    features = _features(model, batches, device)
    probes = {
        name: language_probe(values, sign_languages, videos) for name, values in features.items()
    }
    return EvaluationReport(measures, decision, tests, probes)

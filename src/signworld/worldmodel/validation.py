"""Retrieval on the held-out channel split during training: the metric that decides (§4.10).

Queries are the captions ẽ and the gallery the clips ŷ of a fixed subset of held-out channels;
clips with the same caption count as matches. With K hypotheses (ESP-4) a clip scores its best
hypothesis, as ``F_sem = min_k E_k`` asks. The metric that decides early stopping is the mean
of R@1 text-to-video and video-to-text.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from ..metrics.retrieval import grouped_relevance, recall_at_k
from .distributed import SINGLE, Distributed
from .model import WorldSign, WorldSignBatch

KS = (1, 5, 10)


@dataclass(frozen=True, slots=True)
class RetrievalScores:
    t2v: dict[int, float]
    v2t: dict[int, float]
    clips: int

    @property
    def decision(self) -> float:
        """Mean of R@1 text-to-video and video-to-text."""
        return (self.t2v[1] + self.v2t[1]) / 2

    def as_log(self) -> dict[str, float]:
        scores = {f"t2v_r{k}": v for k, v in self.t2v.items()}
        scores |= {f"v2t_r{k}": v for k, v in self.v2t.items()}
        return scores | {"decision": self.decision, "clips": float(self.clips)}


def similarity(predicted: Tensor, text: Tensor) -> Tensor:
    """(texts, clips) cosine of each caption with each clip's best hypothesis."""
    unit_text = functional.normalize(text.float(), dim=-1)
    unit_video = functional.normalize(predicted.float(), dim=-1)
    return torch.einsum("td,vkd->tvk", unit_text, unit_video).amax(dim=-1)


@torch.no_grad()
def validate(
    model: WorldSign,
    batches: Iterable[WorldSignBatch],
    device: torch.device,
    collective: Distributed = SINGLE,
    *,
    bf16: bool = True,
) -> RetrievalScores:
    """R@k over the validation clips of every GPU."""
    was_training = model.training
    model.eval()
    predicted, texts, rows = [], [], []
    for batch in batches:
        clips = batch.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            predicted.append(model.video.semantic(clips.frames).float().cpu())
            texts.append(model.text(clips.captions, clips.languages).float().cpu())
        if clips.caption_rows is None:
            raise ValueError("validation needs the caption row of every clip")
        rows.append(clips.caption_rows.cpu())
    model.train(was_training)
    gathered = collective.gather_objects(
        [(torch.cat(predicted), torch.cat(texts), torch.cat(rows))]
    )
    all_predicted = torch.cat([part[0] for part in gathered])
    all_texts = torch.cat([part[1] for part in gathered])
    all_rows = torch.cat([part[2] for part in gathered]).tolist()
    scores = similarity(all_predicted, all_texts)
    relevance = grouped_relevance(all_rows, all_rows)
    return RetrievalScores(
        t2v=recall_at_k(scores, relevance, KS),
        v2t=recall_at_k(scores.T, relevance.T, KS),
        clips=len(all_rows),
    )

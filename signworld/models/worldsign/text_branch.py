"""The text side of WorldSign: the target ẽ of the semantic level (§4.4.4).

EmbeddingGemma-300M is frozen and pre-computed (``text.embedding``): the branch receives its
768-dimensional rows, whole, with no truncation. ``LanguageCentering`` subtracts the mean of
the caption's language, ``e° = e - μ_l``, with the means computed once on the training clips;
the branch scales ``e°`` by √768, to about unit variance per dimension. ``TextHead`` is the
two-layer MLP ``768 → 512 → 256`` with GELU and PyTorch's standard initialisation, followed by
a BatchNorm (``batch_norm``), the only trainable part of the branch.
"""

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from signworld.experiment.train.config import TextSettings


class LanguageCentering(nn.Module):
    """``e - μ_l``, one mean per caption language, fitted once on the training clips."""

    means: Tensor

    def __init__(self, width: int) -> None:
        super().__init__()
        self.languages: tuple[str, ...] = ()
        self.register_buffer("means", torch.zeros(0, width))

    def fit(self, embeddings: Tensor, languages: Sequence[str]) -> None:
        """Means of ``embeddings`` (clips, width) per language, from training clips only (§4.6)."""
        if len(embeddings) != len(languages):
            raise ValueError(f"{len(embeddings)} embeddings for {len(languages)} languages")
        names = tuple(sorted(set(languages)))
        labels = torch.tensor([names.index(name) for name in languages])
        rows = embeddings.double()
        means = torch.stack([rows[labels == index].mean(dim=0) for index in range(len(names))])
        self.languages = names
        self.means = means.to(self.means.device, torch.float32)

    def index(self, languages: Sequence[str]) -> Tensor:
        """(clips,) long: each language's row in ``means``; a language never fitted is an error."""
        unknown = sorted(set(languages) - set(self.languages))
        if unknown:
            raise KeyError(f"no mean for languages {unknown}; fitted: {list(self.languages)}")
        return torch.tensor([self.languages.index(name) for name in languages])

    def forward(self, embeddings: Tensor, language: Tensor) -> Tensor:
        """``embeddings`` (batch, width) minus the mean of each row's ``language`` (batch,)."""
        if not self.languages:
            raise RuntimeError("fit the language means on the training clips first")
        centred: Tensor = embeddings.float() - self.means[language]
        return centred

    def get_extra_state(self) -> dict[str, Any]:
        return {"languages": list(self.languages)}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.languages = tuple(state["languages"])

    def _load_from_state_dict(  # type: ignore[no-untyped-def]
        self, state_dict, prefix, *args, **kwargs
    ) -> None:
        """Take the saved number of languages before loading the means."""
        saved = state_dict.get(prefix + "means")
        if saved is not None:
            self.means = torch.empty_like(saved, device=self.means.device)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)  # type: ignore[no-untyped-call]


class TextHead(nn.Sequential):
    """``768 → 512 → 256``, GELU, dropout between the layers, standard initialisation, then a
    BatchNorm that keeps ẽ at unit scale per dimension (``TextSettings.batch_norm``)."""

    def __init__(self, settings: TextSettings) -> None:
        layers: list[nn.Module] = [
            nn.Linear(settings.input_dim, settings.hidden),
            nn.GELU(),
            nn.Dropout(settings.dropout),
            nn.Linear(settings.hidden, settings.output_dim),
        ]
        if settings.batch_norm:
            layers.append(nn.BatchNorm1d(settings.output_dim))
        super().__init__(*layers)


class TextBranch(nn.Module):
    """ẽ(c) = Head(√768 · centring(EmbeddingGemma(c))), from the pre-computed rows."""

    def __init__(self, settings: TextSettings) -> None:
        super().__init__()
        self.centering = LanguageCentering(settings.input_dim)
        self.head = TextHead(settings)
        # NOTE: EmbeddingGemma's rows have unit norm, about 0.036 per dimension, while the
        # head's standard initialisation expects inputs of order 1. Unscaled, the head's bias
        # dominated ẽ at initialisation (pair cosine 0.96, spread 0.005: the collapse of the
        # OpenASL trial v4), and the variance of its outputs (≈2e-5) was close to BatchNorm's
        # ε (1e-5), so ẽ started at std 0.87, not 1. A uniform √768 brings the inputs to about
        # unit variance per dimension (0.9 on OpenASL, after the centring) and leaves their
        # geometry unchanged.
        self.input_scale = math.sqrt(settings.input_dim)

    def forward(self, embeddings: Tensor, language: Tensor) -> Tensor:
        """(batch, 768) EmbeddingGemma rows and (batch,) language indices to ẽ (batch, 256)."""
        centred = self.centering(embeddings, language) * self.input_scale
        target: Tensor = self.head(centred)
        return target

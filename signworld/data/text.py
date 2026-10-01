"""Caption embeddings with a pinned model, revision and prompt (§4.4.4; collaudo §4.13.1).

EmbeddingGemma accepts a task prefix before the text, and a different prefix changes the
geometry without any visible error. The prefix is therefore part of the embedding identity:
model, resolved revision and prompt are hashed into a fingerprint stored with the embeddings,
and every consumer checks that fingerprint. An empty prompt encodes the caption as it is,
which is what sentence-transformers does by default (``default_prompt_name`` is null).
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Protocol, Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import PositiveInt

from ..acquisition.config import FrozenModel

logger = logging.getLogger(__name__)


class EmbeddingSettings(FrozenModel):
    model_id: str = "google/embeddinggemma-300m"
    revision: str = "main"
    """Branch, tag or commit; the commit it resolves to is what gets recorded."""
    prompt: str = ""
    """Task prefix prepended to every caption; empty means the model's plain input."""
    batch_size: PositiveInt = 256


class TextEncoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    model_id: str
    commit: str
    prompt: str

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


class SentenceTransformerEncoder:
    """EmbeddingGemma through sentence-transformers, with the prompt prepended to every text."""

    def __init__(self, settings: EmbeddingSettings, device: str | None = None) -> None:
        from huggingface_hub import model_info
        from sentence_transformers import SentenceTransformer

        commit = model_info(settings.model_id, revision=settings.revision).sha
        if commit is None:
            raise RuntimeError(f"{settings.model_id}@{settings.revision}: commit not resolvable")
        self.identity = EmbeddingIdentity(settings.model_id, commit, settings.prompt)
        self._batch_size = settings.batch_size
        self._model = SentenceTransformer(settings.model_id, revision=commit, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            prompt=self.identity.prompt or None,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


class EmbeddingStore:
    """Embeddings of the unique captions of one manifest, under their fingerprint.

    ``<root>/<fingerprint[:16]>/`` holds ``embeddings.npy`` (one row per unique caption),
    ``captions.parquet`` (the caption of each row), ``clips.parquet`` (clip ID to row) and
    ``identity.json``.
    """

    IDENTITY_FILE: Final = "identity.json"

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @classmethod
    def for_identity(cls, root: Path, identity: EmbeddingIdentity) -> Self:
        return cls(root / identity.fingerprint[:16])

    @property
    def identity(self) -> EmbeddingIdentity:
        raw = json.loads((self.directory / self.IDENTITY_FILE).read_text(encoding="utf-8"))
        return EmbeddingIdentity(raw["model_id"], raw["commit"], raw["prompt"])

    def embeddings(self) -> np.ndarray:
        embeddings: np.ndarray = np.load(self.directory / "embeddings.npy", mmap_mode="r")
        return embeddings

    def captions(self) -> list[str]:
        table = pq.read_table(self.directory / "captions.parquet")
        captions: list[str] = table.column("caption").to_pylist()
        return captions

    def clip_rows(self) -> pa.Table:
        return pq.read_table(self.directory / "clips.parquet")

    def write(
        self,
        identity: EmbeddingIdentity,
        captions: list[str],
        embeddings: np.ndarray,
        clip_ids: list[str],
        rows: list[int],
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(self.directory / "embeddings.npy", embeddings.astype(np.float32))
        pq.write_table(pa.table({"caption": captions}), self.directory / "captions.parquet")
        pq.write_table(
            pa.table({"clip_id": clip_ids, "row": pa.array(rows, pa.int64())}),
            self.directory / "clips.parquet",
        )
        payload = {**asdict(identity), "fingerprint": identity.fingerprint}
        (self.directory / self.IDENTITY_FILE).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def embed_manifest(
    manifest: pa.Table, encoder: TextEncoder, identity: EmbeddingIdentity, root: Path
) -> EmbeddingStore:
    """Embed each distinct caption of the manifest once and map every clip to its row."""
    clip_ids = manifest.column("clip_id").to_pylist()
    captions = manifest.column("caption").to_pylist()
    unique = list(dict.fromkeys(captions))
    row_of = {caption: row for row, caption in enumerate(unique)}
    logger.info("Embedding %d unique captions of %d clips", len(unique), len(captions))
    store = EmbeddingStore.for_identity(root, identity)
    store.write(
        identity,
        unique,
        encoder.encode(unique),
        clip_ids,
        [row_of[caption] for caption in captions],
    )
    return store


@dataclass(frozen=True, slots=True)
class ReproductionReport:
    fingerprint_matches: bool
    sampled: int
    min_cosine: float
    passed: bool


def verify_reproduction(
    store: EmbeddingStore,
    encoder: TextEncoder,
    identity: EmbeddingIdentity,
    *,
    sample: int = 100,
    min_cosine: float = 0.9999,
    seed: int = 0,
) -> ReproductionReport:
    """Collaudo "prompt di EmbeddingGemma": same fingerprint, and re-encoding reproduces rows."""
    stored = store.embeddings()
    rows = np.random.default_rng(seed).choice(len(stored), min(sample, len(stored)), replace=False)
    captions = store.captions()
    fresh = encoder.encode([captions[row] for row in rows])
    reference = stored[rows]
    cosine = (fresh * reference).sum(axis=1) / (
        np.linalg.norm(fresh, axis=1) * np.linalg.norm(reference, axis=1)
    )
    matches = store.identity.fingerprint == identity.fingerprint
    lowest = float(cosine.min())
    return ReproductionReport(matches, len(rows), lowest, matches and lowest > min_cosine)

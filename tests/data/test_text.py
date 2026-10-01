"""Collaudo "prompt di EmbeddingGemma" with a deterministic stand-in encoder."""

import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa

from signworld.data.text import (
    EmbeddingIdentity,
    EmbeddingStore,
    embed_manifest,
    verify_reproduction,
)

IDENTITY = EmbeddingIdentity("stand-in", "0" * 40, "title: none | text: ")


class _HashEncoder:
    """Maps each (prompt, text) to a fixed pseudo-random unit vector, and counts its calls."""

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.encoded: list[str] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.encoded.extend(texts)
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256((self.prompt + text).encode()).digest()[:8])
            vector = np.random.default_rng(seed).standard_normal(32)
            rows.append(vector / np.linalg.norm(vector))
        return np.asarray(rows, dtype=np.float32)


def _manifest() -> pa.Table:
    return pa.table(
        {"clip_id": ["a", "b", "c", "d"], "caption": ["Hello!", "Rain today.", "Hello!", "Bye."]}
    )


def test_each_unique_caption_is_embedded_once_and_clips_map_to_rows(tmp_path: Path) -> None:
    encoder = _HashEncoder(IDENTITY.prompt)

    store = embed_manifest(_manifest(), encoder, IDENTITY, tmp_path)

    assert encoder.encoded == ["Hello!", "Rain today.", "Bye."]
    assert store.clip_rows().column("row").to_pylist() == [0, 1, 0, 2]
    assert store.embeddings().shape == (3, 32)
    assert store.identity == IDENTITY


def test_same_encoder_and_prompt_reproduce_the_store(tmp_path: Path) -> None:
    store = embed_manifest(_manifest(), _HashEncoder(IDENTITY.prompt), IDENTITY, tmp_path)

    report = verify_reproduction(store, _HashEncoder(IDENTITY.prompt), IDENTITY)

    assert report.passed
    assert report.sampled == 3


def test_a_different_prompt_fails_both_fingerprint_and_cosine(tmp_path: Path) -> None:
    store = embed_manifest(_manifest(), _HashEncoder(IDENTITY.prompt), IDENTITY, tmp_path)
    other = EmbeddingIdentity(IDENTITY.model_id, IDENTITY.commit, "task: search result | query: ")

    report = verify_reproduction(store, _HashEncoder(other.prompt), other)

    assert not report.fingerprint_matches
    assert report.min_cosine < 0.9
    assert not report.passed


def test_fingerprint_changes_with_any_part_of_the_identity() -> None:
    fingerprints = {
        IDENTITY.fingerprint,
        EmbeddingIdentity("other", IDENTITY.commit, IDENTITY.prompt).fingerprint,
        EmbeddingIdentity(IDENTITY.model_id, "1" * 40, IDENTITY.prompt).fingerprint,
        EmbeddingIdentity(IDENTITY.model_id, IDENTITY.commit, "title: x | text: ").fingerprint,
    }

    assert len(fingerprints) == 4
    assert (
        EmbeddingStore.for_identity(Path("/r"), IDENTITY).directory.name
        == (IDENTITY.fingerprint[:16])
    )

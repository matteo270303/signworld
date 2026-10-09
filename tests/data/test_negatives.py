"""GloFND's thresholds, fixed on the frozen captions (arm C)."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import torch

from signworld.data.negatives import UNUSED, compute_thresholds, load_thresholds, write_thresholds
from signworld.loss.worldsign import FalseNegatives


def _corpus(rows: int = 40, seed: int = 0) -> tuple[np.ndarray, pa.Table]:
    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((rows + 2, 16)).astype(np.float32)
    embeddings[1] = embeddings[0] + 0.01 * rng.standard_normal(16)  # a near duplicate
    train = pa.table(
        {
            "caption_row": list(range(rows)),
            "caption_language": ["en"] * (rows // 2) + ["de"] * (rows - rows // 2),
        }
    )
    return embeddings, train


def test_each_caption_drops_its_top_alpha_share_of_the_reference() -> None:
    embeddings, train = _corpus()

    thresholds, anchors, rank = compute_thresholds(embeddings, train, alpha=0.05, reference=40)

    assert anchors == 40 and rank == 2  # ceil(0.05 · 40)
    assert (thresholds[40:] == UNUSED).all()  # rows no training caption uses
    vectors = torch.from_numpy(embeddings[:40])
    centred = vectors.clone()
    centred[:20] -= vectors[:20].mean(0)
    centred[20:] -= vectors[20:].mean(0)
    mask = FalseNegatives(torch.from_numpy(thresholds)).mask(centred, torch.arange(40))
    assert mask[0, 1] and mask[1, 0]  # the near duplicate is among the top two of both
    flagged_by_row = (
        torch.nn.functional.normalize(centred, dim=-1)
        @ torch.nn.functional.normalize(centred, dim=-1).T
    ).fill_diagonal_(-2) >= torch.from_numpy(thresholds[:40])[:, None] - 1e-6
    assert flagged_by_row.sum(1).tolist() == [rank] * 40


def test_thresholds_round_trip_and_check_the_run_s_alpha(tmp_path: Path) -> None:
    embeddings, train = _corpus()
    index = tmp_path / "index.parquet"
    index.write_bytes(b"index")

    meta = write_thresholds(
        tmp_path / "negatives", embeddings, train, store="abc", index=index, alpha=0.05
    )

    assert meta.rows == len(embeddings) and meta.rank == 2 and meta.store == "abc"
    loaded = load_thresholds(tmp_path / "negatives", 0.05, rows=len(embeddings))
    assert loaded.shape == (len(embeddings),)
    with pytest.raises(ValueError, match="alpha"):
        load_thresholds(tmp_path / "negatives", 1e-3)

"""PC1 on synthetic embeddings with a known intrinsic dimension."""

import numpy as np
import pytest
import torch

from signworld.experiment.collaudo.text_geometry import (
    GeometryAtDimension,
    TextGeometrySettings,
    center_by_language,
    choose_dimension,
    language_variance_fraction,
    text_geometry,
)

SETTINGS = TextGeometrySettings(
    dimensions=(64, 32, 16, 8), sigreg_samples=2048, sigreg_directions=64, cosine_pairs=10_000
)


def _row(dimension: int, recall: float) -> GeometryAtDimension:
    return GeometryAtDimension(
        dimension, float(dimension), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [], neighbour_recall=recall
    )


def test_chosen_dimension_is_the_smallest_keeping_the_neighbours() -> None:
    rows = [_row(768, 1.0), _row(512, 0.93), _row(256, 0.88), _row(128, 0.95)]

    assert choose_dimension(rows, retained_neighbours=0.9) == 128


def test_the_old_rank_rule_would_have_been_blind_to_truncation() -> None:
    generator = np.random.default_rng(3)
    isotropic = generator.standard_normal((4000, 64)).astype(np.float32)

    report = text_geometry(isotropic, ["en"] * 4000, SETTINGS)

    # Every truncation of isotropic data keeps nearly all its rank relative to its own size,
    # but loses the neighbours: only the full dimension may be chosen.
    recalls = {row.dimension: row.neighbour_recall for row in report.per_dimension}
    assert recalls[64] == 1.0
    assert recalls[8] < 0.5
    assert report.chosen_dimension == 64


def test_centering_removes_each_language_mean() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 5.0], [0.0, 7.0]])

    centred = center_by_language(embeddings, ["en", "en", "de", "de"])

    assert torch.allclose(centred, torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]))


def test_language_offsets_do_not_inflate_the_random_pair_cosine() -> None:
    generator = np.random.default_rng(0)
    base = generator.standard_normal((4000, 64))
    offsets = np.where(np.arange(4000)[:, None] < 2000, 3.0, -3.0) * np.eye(64)[0]
    languages = ["en" if index < 2000 else "de" for index in range(4000)]

    report = text_geometry((base + offsets).astype(np.float32), languages, SETTINGS)

    full = report.per_dimension[0]
    assert abs(full.mean_random_pair_cosine) < 0.05
    assert full.language_variance_fraction > 0.05
    assert report.languages == {"de": 2000, "en": 2000}


def test_matryoshka_like_targets_choose_a_small_dimension_and_score_anisotropic() -> None:
    generator = np.random.default_rng(1)
    # As in Matryoshka training, the leading coordinates carry the signal.
    low_rank = np.concatenate(
        [generator.standard_normal((4000, 6)), 0.02 * generator.standard_normal((4000, 58))],
        axis=1,
    )
    isotropic = generator.standard_normal((4000, 64))

    low = text_geometry(low_rank.astype(np.float32), ["en"] * 4000, SETTINGS)
    iso = text_geometry(isotropic.astype(np.float32), ["en"] * 4000, SETTINGS)

    assert low.chosen_dimension < iso.chosen_dimension
    assert low.per_dimension[0].isoscore < iso.per_dimension[0].isoscore
    assert low.per_dimension[0].sigreg > iso.per_dimension[0].sigreg
    assert iso.per_dimension[0].sigreg < 3 * iso.per_dimension[0].sigreg_gaussian_reference


def test_requesting_more_dimensions_than_available_fails() -> None:
    with pytest.raises(ValueError, match="fewer than requested"):
        text_geometry(np.zeros((10, 16), dtype=np.float32), ["en"] * 10, SETTINGS)


def test_language_share_is_zero_without_language_offsets() -> None:
    embeddings = torch.randn(1000, 8, generator=torch.Generator().manual_seed(0))

    share = language_variance_fraction(embeddings, ["en"] * 500 + ["de"] * 500)

    assert share < 0.01

"""The spectrum finds how many directions a latent uses, and whitens only those."""

from pathlib import Path

import numpy as np

from signworld.experiments.pose_spectrum import PrincipalAxes, save_whitenings, spectrum
from signworld.experiments.pose_teachers import PoseCorpus
from signworld.metrics.probes import video_split
from signworld.pose.tokens import STEPS, articulator_columns
from signworld.pose.wholebody import Articulator

from .test_pose_teachers import TINY, _clips


def test_whitening_gives_identity_covariance_on_its_rows() -> None:
    rng = np.random.default_rng(0)
    rows = rng.standard_normal((5000, 6)) @ rng.standard_normal((6, 6)) * [1, 2, 3, 4, 5, 6]

    whitened = PrincipalAxes.of(rows).whitening(4).apply(rows).astype(np.float64)

    np.testing.assert_allclose(np.cov(whitened, rowvar=False), np.eye(4), atol=1e-3)


def test_a_latent_that_is_a_few_directions_is_cut_there(tmp_path: Path) -> None:
    corpus = PoseCorpus.load(_clips(tmp_path, videos=40), TINY.min_score)
    test = video_split(corpus.videos)
    rng = np.random.default_rng(1)
    features = {}
    for part in Articulator:
        # The latent is a linear image of the positions in 256 dimensions, plus faint noise.
        positions = corpus.positions[:, :, articulator_columns(part)].reshape(
            len(corpus.clips), STEPS, -1
        )
        mixing = rng.standard_normal((positions.shape[-1], 256))
        noise = 1e-3 * rng.standard_normal((len(corpus.clips), STEPS, 256))
        features[part] = (positions @ mixing + noise).astype(np.float32)

    report, whitenings = spectrum(features, corpus, test, grid=(8, 16, 48, 64, 256))

    hand = report.articulators["right"]  # 21 keypoints: 42 coordinates
    assert hand.directions_for_99 <= 42
    assert hand.chosen_k == 48
    chosen = next(t for t in hand.truncations if t.k == 48)
    assert chosen.isoscore_whitened > hand.isoscore
    assert whitenings[Articulator.RIGHT_HAND].projection.shape == (256, 48)
    saved = np.load(save_whitenings(whitenings, tmp_path / "whitening.npz"))
    assert saved["right_projection"].shape == (256, 48)

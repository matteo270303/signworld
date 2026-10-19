"""The text branch: centring per language and the MLP head (§4.4.4)."""

import pytest
import torch

from signworld.experiment.train.config import TextSettings
from signworld.models.worldsign.text_branch import LanguageCentering, TextBranch


def _captions() -> tuple[torch.Tensor, list[str]]:
    generator = torch.Generator().manual_seed(0)
    embeddings = torch.randn(40, 768, generator=generator)
    languages = ["en"] * 25 + ["es"] * 15
    embeddings[25:] += 2.0  # the Spanish captions sit elsewhere
    return embeddings, languages


def test_centred_captions_have_zero_mean_in_every_language() -> None:
    """P11: target text embeddings at mean ≈ 0 per language."""
    embeddings, languages = _captions()
    centering = LanguageCentering(768)
    centering.fit(embeddings, languages)
    index = centering.index(languages)

    centred = centering(embeddings, index)

    assert centering.languages == ("en", "es")
    for name in ("en", "es"):
        rows = centred[[language == name for language in languages]]
        assert rows.mean(dim=0).abs().max() < 1e-5


def test_a_language_never_fitted_and_an_unfitted_centering_are_errors() -> None:
    embeddings, languages = _captions()
    centering = LanguageCentering(768)

    with pytest.raises(RuntimeError, match="fit"):
        centering(embeddings, torch.zeros(40, dtype=torch.long))
    centering.fit(embeddings, languages)
    with pytest.raises(KeyError, match="de"):
        centering.index(["en", "de"])


def test_the_means_and_their_languages_survive_a_checkpoint() -> None:
    embeddings, languages = _captions()
    fitted = LanguageCentering(768)
    fitted.fit(embeddings, languages)

    restored = LanguageCentering(768)
    restored.load_state_dict(fitted.state_dict())

    assert restored.languages == fitted.languages
    assert torch.equal(restored.means, fitted.means)


def test_the_head_is_768_512_256_with_gelu_and_a_batch_norm_and_trains() -> None:
    branch = TextBranch(TextSettings())
    embeddings, languages = _captions()
    branch.centering.fit(embeddings, languages)

    target = branch(embeddings, branch.centering.index(languages))
    target.pow(2).mean().backward()

    assert target.shape == (40, 256)
    assert (
        sum(p.numel() for p in branch.parameters() if p.requires_grad)
        == 768 * 512 + 512 + 512 * 256 + 256 + 2 * 256
    )  # §4.8, and the BatchNorm's scale and shift
    assert [type(m).__name__ for m in branch.head] == [
        "Linear",
        "GELU",
        "Dropout",
        "Linear",
        "BatchNorm1d",
    ]
    assert all(p.grad is not None for p in branch.head.parameters())
    assert not branch.centering.means.requires_grad


def test_the_target_starts_at_unit_scale_whatever_the_size_of_the_captions() -> None:
    """Unit-norm EmbeddingGemma rows (≈0.04 per dimension) used to give a target dominated by
    the head's bias and below BatchNorm's ε: scaled by √768 and batch-normalised, ẽ starts at
    unit scale."""
    embeddings, languages = _captions()
    rows = torch.nn.functional.normalize(embeddings, dim=-1)
    branch = TextBranch(TextSettings())
    branch.centering.fit(rows, languages)
    index = branch.centering.index(languages)

    with torch.no_grad():
        inputs = branch.centering(rows, index) * branch.input_scale
        target = branch(rows, index)

    assert 0.5 < inputs.std(dim=0).mean() < 1.5  # of order one (the centring removes a share)
    assert (inputs / branch.input_scale).std(dim=0).mean() < 0.05  # what the head used to see
    assert target.std(dim=0).mean() == pytest.approx(1.0, rel=0.05)

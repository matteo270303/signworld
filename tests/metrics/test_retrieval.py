"""Collaudo "testare i test" (§4.13.1): retrieval metrics on cases with a known answer."""

import pytest
import torch

from signworld.metrics.retrieval import (
    bootstrap_recall,
    grouped_relevance,
    hubness,
    match_ranks,
    paired_relevance,
    recall_at_k,
)


def _cosine(queries: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
    return (
        torch.nn.functional.normalize(queries, dim=1)
        @ torch.nn.functional.normalize(gallery, dim=1).T
    )


def test_identical_embeddings_give_perfect_recall() -> None:
    embeddings = torch.randn(500, 64, generator=torch.Generator().manual_seed(0))

    recall = recall_at_k(_cosine(embeddings, embeddings), paired_relevance(500), [1, 5])

    assert recall == {1: 1.0, 5: 1.0}


def test_random_embeddings_give_chance_recall() -> None:
    generator = torch.Generator().manual_seed(0)
    size = 2000
    similarity = _cosine(torch.randn(size, 64, generator=generator), torch.randn(size, 64))

    recall = recall_at_k(similarity, paired_relevance(size), [1, 10, 100])

    for k, value in recall.items():
        expected = k / size
        tolerance = 4 * (expected * (1 - expected) / size) ** 0.5
        assert value == pytest.approx(expected, abs=tolerance)


def test_constant_embeddings_do_not_win_through_ties() -> None:
    constant = torch.ones(100, 8)

    recall = recall_at_k(_cosine(constant, constant), paired_relevance(100), [1, 99, 100])

    assert recall == {1: 0.0, 99: 0.0, 100: 1.0}


def test_duplicate_tolerant_recall_accepts_an_equivalent_item() -> None:
    similarity = torch.tensor([[0.2, 0.9, 0.1], [0.9, 0.3, 0.0], [0.0, 0.1, 0.8]])
    groups = ["hello", "hello", "rain"]

    standard = recall_at_k(similarity, paired_relevance(3), [1])[1]
    tolerant = recall_at_k(similarity, grouped_relevance(groups, groups), [1])[1]

    assert standard == pytest.approx(1 / 3)
    assert tolerant == 1.0


def test_queries_without_a_relevant_item_are_rejected() -> None:
    with pytest.raises(ValueError, match="relevant"):
        match_ranks(torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.bool))


def test_bootstrap_interval_contains_the_estimate() -> None:
    generator = torch.Generator().manual_seed(0)
    similarity = torch.randn(300, 300, generator=generator) + 3 * torch.eye(300)

    interval = bootstrap_recall(
        similarity, paired_relevance(300), 1, generator=torch.Generator().manual_seed(1)
    )

    assert interval.low <= interval.estimate <= interval.high
    assert interval.high - interval.low < 0.2


def test_hubness_is_high_when_one_item_attracts_every_query() -> None:
    generator = torch.Generator().manual_seed(0)
    balanced = torch.randn(1000, 1000, generator=generator)
    hub = balanced.clone()
    hub[:, 0] = 10.0

    assert hubness(hub, k=5) > 10 * abs(hubness(balanced, k=5))

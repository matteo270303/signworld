"""The gates decide at the boundaries written in §4.12.2."""

import pytest

from signworld.evaluation.gate import Decision, GatePolicy

POLICY = GatePolicy()


@pytest.mark.parametrize(
    ("r1", "expected"),
    [
        (20.0, Decision.STOP_BELOW_BASELINE),
        (31.0, Decision.STOP),
        (31.1, Decision.SECOND_GATE),
        (55.9, Decision.SECOND_GATE),
        (56.0, Decision.PROCEED),
    ],
)
def test_first_gate_boundaries(r1: float, expected: Decision) -> None:
    assert POLICY.first(r1, ridge_baseline=25.0) is expected


def test_second_gate_requires_three_quarters_of_c2rl() -> None:
    assert POLICY.second(46.7)
    assert not POLICY.second(46.6)

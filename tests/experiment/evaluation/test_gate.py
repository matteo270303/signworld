"""The gate decides at the boundaries written in §4.12.2: each direction, the worse decides."""

import pytest

from signworld.experiment.evaluation.gate import Decision, GatePolicy
from signworld.metrics.directions import Bidirectional

POLICY = GatePolicy()
RIDGE = Bidirectional(t2v=25.0, v2t=25.0)


@pytest.mark.parametrize(
    ("t2v", "v2t", "expected"),
    [
        (20.0, 40.0, Decision.STOP_BELOW_BASELINE),  # one direction under its ridge baseline
        (40.0, 24.9, Decision.STOP_BELOW_BASELINE),
        (31.0, 50.0, Decision.STOP),  # half of C²RL: 31.1 text → video ...
        (50.0, 30.7, Decision.STOP),  # ... and 30.8 video → text
        (31.1, 30.8, Decision.PROCEED),
        (56.0, 56.0, Decision.PROCEED),
    ],
)
def test_final_gate_boundaries(t2v: float, v2t: float, expected: Decision) -> None:
    assert POLICY.final(Bidirectional(t2v=t2v, v2t=v2t), ridge_baseline=RIDGE) is expected


def test_stop_f3_requires_three_quarters_of_c2rl_mean() -> None:
    assert POLICY.on_track(46.5)
    assert not POLICY.on_track(46.4)

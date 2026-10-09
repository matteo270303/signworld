import time

import pytest

from signworld.data.acquisition.config import RefusalPolicy
from signworld.data.acquisition.outcome import Status
from signworld.data.acquisition.pacing import DeadlineExceededError, RefusalBackoff, deadline


def test_deadline_interrupts_a_stalled_block() -> None:
    with pytest.raises(DeadlineExceededError), deadline(0.05):
        time.sleep(5)


def test_deadline_cannot_be_swallowed_by_broad_exception_handlers() -> None:
    def stubborn() -> None:
        try:
            time.sleep(5)
        except Exception:
            pytest.fail("the deadline was caught as an ordinary exception")

    with pytest.raises(DeadlineExceededError), deadline(0.05):
        stubborn()


def test_deadline_is_disarmed_after_the_block() -> None:
    with deadline(0.05):
        pass

    time.sleep(0.1)


def test_backoff_gives_up_at_the_maximum_streak() -> None:
    pauses: list[float] = []
    backoff = RefusalBackoff(RefusalPolicy(max_consecutive=4, cooldown_s=1.0), pauses.append)

    decisions = [backoff.should_continue(Status.BLOCKED) for _ in range(4)]

    assert decisions == [True, True, True, False]
    assert pauses == [1.0, 2.0, 4.0]

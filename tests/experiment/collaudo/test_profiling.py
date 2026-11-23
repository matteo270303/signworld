import torch

from signworld.experiment.collaudo.profiling import Clock


class Counter:
    def __init__(self) -> None:
        self.calls = 0

    def work(self, value: int) -> int:
        self.calls += 1
        return value * 2


def test_sections_nest_and_count_every_call_of_a_step() -> None:
    clock = Clock(torch.device("cpu"))
    owner = Counter()
    clock.wrap(owner, "work", "work")
    warmup, steps = 1, 3
    for _ in range(warmup + steps):
        with clock.section("step"):
            assert owner.work(2) == 4  # the wrapped method still answers
            owner.work(3)  # twice per step, as the encoder once per mask kind
    assert owner.calls == 2 * (warmup + steps)
    assert set(clock.seconds) == {"step", "step/work"}
    seconds = clock.per_step(warmup, steps)
    assert seconds["step/work"] <= seconds["step"]
    assert len(clock.seconds["step/work"]) == 2 * (warmup + steps)

"""The gate (§4.12.2), written down before the training loop so no result can move it.

It reads R@1 in both directions, text → video and video → text, in percent, on the OpenASL test
split **without fine-tuning**, each against its own reference: C²RL, which reaches 62.2 and 61.6
**with** fine-tuning, a ResNet-18 at 224² and other data [Lett. 87] (a sanity check, not a
claim), and the ridge baseline of PC2 in the same direction. A direction that fails decides.

* The gate run is run 1 of the plan (§4.14): ViT-L, arm A (alignment + SIGReg). The other
  four arms of ESP-1 start once it passes stop F2 and stop with it at F3; the ablations on the
  best arm follow ESP-1.
* At stop F3 the extrapolated curve of the metric that decides, the mean R@1 of the two
  directions, must stay compatible with X = 0.75 of C²RL's mean. X was the second gate before
  the ViT-L row of ESP-1; with the ViT-B row gone (revision of 29/9) it only judges F3.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from signworld.metrics.directions import Bidirectional

C2RL_R1: Final = Bidirectional(t2v=62.2, v2t=61.6)
"""C²RL on the OpenASL test split, R@1 in percent, after fine-tuning [Lett. 87]."""
X_R1: Final = 46.5
"""X = 0.75 * 61.9 (the mean of C²RL's two directions) = 46.425, rounded up to the precision
R@1 is reported with (4/10; it was 0.75 * 62.2 = 46.7 on text → video alone)."""


class Decision(StrEnum):
    STOP_BELOW_BASELINE = "stop: below the ridge baseline of PC2, a bug or a harmful objective"
    STOP = "stop: below half of C²RL, beyond what adjustments can recover"
    PROCEED = "proceed: compare the arms and build the final model"


@dataclass(frozen=True, slots=True)
class GatePolicy:
    reference: Bidirectional = C2RL_R1
    stop_below: float = 0.5
    x: float = X_R1

    def final(self, r1: Bidirectional, ridge_baseline: Bidirectional) -> Decision:
        """The gate run's R@1 at F4, in percent, each direction against its own ridge
        baseline of PC2 and half of its own C²RL reference: the worse direction decides."""
        directions = [
            (value, baseline, reference)
            for (_, value), (_, baseline), (_, reference) in zip(
                r1.items(), ridge_baseline.items(), self.reference.items(), strict=True
            )
        ]
        if any(value < baseline for value, baseline, _ in directions):
            return Decision.STOP_BELOW_BASELINE
        if any(value < self.stop_below * reference for value, _, reference in directions):
            return Decision.STOP
        return Decision.PROCEED

    def on_track(self, extrapolated_r1: float) -> bool:
        """Stop F3: whether the extrapolated curve of the mean R@1 of the two directions, in
        percent, stays compatible with X."""
        return extrapolated_r1 >= self.x

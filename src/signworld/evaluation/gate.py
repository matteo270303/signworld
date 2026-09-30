"""The gate (§4.12.2), written down before the training loop so no result can move it.

It reads R@1 text → video, in percent, on the OpenASL test split **without fine-tuning**,
against C²RL, which reaches 62.2 **with** fine-tuning, a ResNet-18 at 224² and other data
[Lett. 87]: a sanity check, not a claim.

* The gate run is run 1 of the plan (§4.14): ViT-L, arm A (alignment + SIGReg). The other
  four arms of ESP-1 start once it passes stop F2 and stop with it at F3; the ablations on the
  best arm follow ESP-1.
* At stop F3 the extrapolated R@1 curve must stay compatible with X = 0.75 of C²RL. X was the
  second gate before the ViT-L row of ESP-1; with the ViT-B row gone (revision of 29/9) it only
  judges F3.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

C2RL_R1_T2V: Final = 62.2
"""C²RL on the OpenASL test split, text → video R@1, after fine-tuning [Lett. 87]."""
X_R1: Final = 46.7
"""X = 0.75 * 62.2 = 46.65, rounded up to the precision R@1 is reported with."""


class Decision(StrEnum):
    STOP_BELOW_BASELINE = "stop: below the ridge baseline of PC2, a bug or a harmful objective"
    STOP = "stop: below half of C²RL, beyond what adjustments can recover"
    PROCEED = "proceed: compare the arms and build the final model"


@dataclass(frozen=True, slots=True)
class GatePolicy:
    reference: float = C2RL_R1_T2V
    stop_below: float = 0.5
    x: float = X_R1

    def final(self, r1: float, ridge_baseline: float) -> Decision:
        """The gate run's R@1 at F4 against the ridge baseline of PC2 and half of C²RL."""
        if r1 < ridge_baseline:
            return Decision.STOP_BELOW_BASELINE
        if r1 < self.stop_below * self.reference:
            return Decision.STOP
        return Decision.PROCEED

    def on_track(self, extrapolated_r1: float) -> bool:
        """Stop F3: whether the extrapolated R@1 curve stays compatible with X."""
        return extrapolated_r1 >= self.x

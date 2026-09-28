"""The phase-1 gates (§4.12.2), written down before the training loop so no result can move them.

Both read R@1 text → video, in percent, on the OpenASL test split **without fine-tuning**,
against C²RL, which reaches 62.2 **with** fine-tuning, a ResNet-18 at 224² and other data
[Lett. 87]: a sanity check, not a claim.

* The first gate judges the gate run (ASL only, ViT-B, arm A).
* The second gate applies only when the first lands between 0.5 and 0.9 of C²RL. It sits before
  the ViT-L row of ESP-1, about 43 % of the whole budget (§4.14): the best ViT-B arm of ESP-1
  must reach X = 0.75 of C²RL, else the ViT-L row does not start.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

C2RL_R1_T2V: Final = 62.2
"""C²RL on the OpenASL test split, text → video R@1, after fine-tuning [Lett. 87]."""
SECOND_GATE_R1: Final = 46.7
"""X = 0.75 * 62.2 = 46.65, rounded up to the precision R@1 is reported with."""


class Decision(StrEnum):
    STOP_BELOW_BASELINE = "stop: below the ridge baseline of PC2, a bug or a harmful objective"
    STOP = "stop: below half of C²RL, beyond what adjustments can recover"
    SECOND_GATE = "proceed to the ViT-B row of ESP-1; the second gate decides the ViT-L row"
    PROCEED = "proceed"


@dataclass(frozen=True, slots=True)
class GatePolicy:
    reference: float = C2RL_R1_T2V
    stop_below: float = 0.5
    proceed_from: float = 0.9
    second_gate: float = SECOND_GATE_R1

    def first(self, r1: float, ridge_baseline: float) -> Decision:
        """The gate run's R@1 against the ridge baseline of PC2 and the share of C²RL."""
        if r1 < ridge_baseline:
            return Decision.STOP_BELOW_BASELINE
        if r1 < self.stop_below * self.reference:
            return Decision.STOP
        if r1 < self.proceed_from * self.reference:
            return Decision.SECOND_GATE
        return Decision.PROCEED

    def second(self, best_vit_b_r1: float) -> bool:
        """Whether the ViT-L row of ESP-1 may start, from its best ViT-B arm."""
        return best_vit_b_r1 >= self.second_gate

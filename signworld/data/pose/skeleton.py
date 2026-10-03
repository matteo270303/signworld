"""The skeleton of the 69 joints the pose encoder reads: one part per articulator (posa §2).

A joint's local position is measured from its part's root (the wrist of a hand, the nose tip
of the face; the body has none, the origin between the shoulders being its reference) and its
bone goes to its parent along the part's chains, as in worldSign's pose encoder. ``PARENT`` and
``ROOT`` use COCO-WholeBody indices; ``Part`` stores columns of ``JOINTS``, where every
articulator's joints are contiguous.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from .tokens import JOINTS
from .wholebody import ARTICULATOR_INDICES, LEFT_SHOULDER, RIGHT_SHOULDER, Articulator

NOSE: Final = 0
NOSE_TIP: Final = 53
LEFT_HAND_WRIST: Final = 91
RIGHT_HAND_WRIST: Final = 112
_FINGER_PARENTS: Final = (0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19)
"""Parent of each of a hand's 21 keypoints, relative to its wrist: five four-joint chains."""


def _chain(joints: Sequence[int], anchor: int) -> dict[int, int]:
    """Each joint hangs from the one before it, the first from ``anchor``."""
    return {joint: anchor if i == 0 else joints[i - 1] for i, joint in enumerate(joints)}


def _hand(wrist: int) -> dict[int, int]:
    return {wrist + i: wrist + parent for i, parent in enumerate(_FINGER_PARENTS)}


PARENT: Final[dict[int, int]] = {
    # The body hangs from the nose: ears and shoulders from it, then elbows and wrists.
    NOSE: NOSE,
    3: NOSE,
    4: NOSE,
    LEFT_SHOULDER: NOSE,
    RIGHT_SHOULDER: NOSE,
    7: LEFT_SHOULDER,
    8: RIGHT_SHOULDER,
    9: 7,
    10: 8,
    **_hand(LEFT_HAND_WRIST),
    **_hand(RIGHT_HAND_WRIST),
    **_chain(tuple(range(23, 40, 2)), NOSE_TIP),  # every other jaw point
    **_chain(tuple(range(83, 91)), NOSE_TIP),  # the inner lip
    NOSE_TIP: NOSE_TIP,
}
"""COCO-WholeBody index of every joint's parent; a chain's top is its own parent."""

ROOT: Final[dict[Articulator, int | None]] = {
    Articulator.BODY: None,
    Articulator.LEFT_HAND: LEFT_HAND_WRIST,
    Articulator.RIGHT_HAND: RIGHT_HAND_WRIST,
    Articulator.FACE: NOSE_TIP,
}
"""COCO-WholeBody index of each part's root; None: the origin between the shoulders."""


@dataclass(frozen=True, slots=True)
class Part:
    """One articulator as columns of ``JOINTS``."""

    articulator: Articulator
    start: int
    stop: int
    root: int | None
    """Column of the root joint; None: the origin."""
    parents: tuple[int, ...]
    """Column of each joint's parent, in the part's order."""

    @property
    def size(self) -> int:
        return self.stop - self.start


def _part(articulator: Articulator) -> Part:
    columns = [JOINTS.index(joint) for joint in ARTICULATOR_INDICES[articulator]]
    start, stop = columns[0], columns[-1] + 1
    if columns != list(range(start, stop)):
        raise ValueError(f"the joints of {articulator} are not contiguous in JOINTS")
    parents = tuple(JOINTS.index(PARENT[joint]) for joint in ARTICULATOR_INDICES[articulator])
    if any(not start <= parent < stop for parent in parents):
        raise ValueError(f"a joint of {articulator} hangs from another articulator")
    root = ROOT[articulator]
    return Part(articulator, start, stop, None if root is None else JOINTS.index(root), parents)


PARTS: Final = tuple(_part(articulator) for articulator in Articulator)
"""The four parts, in the order of ``Articulator`` and of ``JOINTS``."""
SHOULDER_COLUMNS: Final = (JOINTS.index(LEFT_SHOULDER), JOINTS.index(RIGHT_SHOULDER))

"""CSL-Daily (Zhou et al., 2021): daily-life Chinese Sign Language, released after an agreement.

The release is JPG frames, one folder per sentence. Once the files are in place (``local_root``),
fetching turns each folder into ``raw/videos/<sentence>.mp4``, the form the pipeline reads.
"""

import logging
import pickle
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from pydantic import PositiveFloat

from ..config import FrozenModel
from ..outcome import Outcome, Status
from ..transfer.frames import encode_frames
from .base import Access, ManualDatasetSource

logger = logging.getLogger(__name__)


class CSLDailySettings(FrozenModel):
    local_root: Path | None = None
    """The release as received; unset until you have it (fetching then explains how)."""
    labels_file: str = "sentence_label/csl2020ct_v2.pkl"
    split_file: str = "sentence_label/split_1.txt"
    frames_dir: str = "data/frames_512x512"
    fps: PositiveFloat = 30.0


@dataclass(frozen=True, slots=True)
class CSLSentence:
    name: str
    frames: int
    caption: str
    """The characters, the SLT target of the release."""
    signer: int
    split: str
    """``train``, ``valid`` or ``test`` (the release calls the second one ``dev``)."""


class CSLDailySource(ManualDatasetSource[CSLDailySettings]):
    name = "csl_daily"
    homepage = "https://ustc-slr.github.io/datasets/2021_csl_daily/"
    terms = "CSL Dataset Release Agreement, signed by a full-time staff member (not a student)."
    access = Access.MANUAL
    settings_model = CSLDailySettings
    instructions = (
        "send the signed CSL Dataset Release Agreement to ustc_vslrg@126.com with "
        "zhwg@ustc.edu.cn in copy (see the homepage), then set sources.csl_daily.local_root to "
        "the folder with the files you receive."
    )

    @property
    def local_root(self) -> Path:
        if self.settings.local_root is None:
            self.check_access()
        assert self.settings.local_root is not None  # noqa: S101 (narrowing after check_access)
        return self.settings.local_root

    def check_access(self) -> None:
        if self.settings.local_root is None:
            super().check_access()

    def fetch_metadata(self) -> None:
        for relative in (self.settings.labels_file, self.settings.split_file):
            self.provenance.register(
                self.local_root / relative, origin=f"local copy of {self.homepage}"
            )

    @cached_property
    def sentences(self) -> list[CSLSentence]:
        """Every sentence with an official split; the one without is left out."""
        with (self.local_root / self.settings.labels_file).open("rb") as stream:
            labels = pickle.load(stream)["info"]  # noqa: S301 (the release's own file)
        split_of: dict[str, str] = {}
        with (self.local_root / self.settings.split_file).open(encoding="utf-8") as stream:
            next(stream)  # header: name|split
            for line in stream:
                name, split = line.strip().split("|")
                split_of[name] = {"dev": "valid"}.get(split, split)
        sentences = []
        for row in labels:
            if row["name"] not in split_of:
                logger.warning("%s: in no official split, left out", row["name"])
                continue
            sentences.append(
                CSLSentence(
                    row["name"],
                    row["length"],
                    "".join(row["label_char"]),
                    row["signer"],
                    split_of[row["name"]],
                )
            )
        return sentences

    def media_keys(self) -> list[str]:
        if self.settings.local_root is None:
            return []
        return [sentence.name for sentence in self.sentences]

    def fetch_item(self, key: str) -> Outcome:
        target = self.layout.raw / "videos" / f"{key}.mp4"
        if target.is_file():
            return Outcome(key, Status.DONE, str(target))
        frames = self.local_root / self.settings.frames_dir / key
        if not frames.is_dir():
            return Outcome(key, Status.UNAVAILABLE, "no frame folder in the local copy")
        encode_frames(frames, self.settings.fps, target, ".jpg")
        return Outcome(key, Status.DONE, str(target))

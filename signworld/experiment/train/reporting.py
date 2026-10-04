"""What a run prints and tabulates, set up as the worldSign runs do it.

* a progress bar per epoch on standard error (``tqdm``, first GPU, refreshed every 15 s) with
  the stage, the epoch's running loss, the R@1 of the latest validation and the expected end
  of the run; a bar over the validation clips;
* one summary line per validation on standard output, and one row with fixed columns in
  ``metrics.csv`` (the epoch ends and the final validation) or ``val_steps.csv`` (the others:
  step 0, the end of the constant phase, ...);
* one line per checkpoint written.

Every reading also stays in ``metrics.jsonl``. Everything here is a no-op off the first GPU.
"""

import csv
import math
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from tqdm import tqdm  # type: ignore[import-untyped]

from signworld.metrics.retrieval import KS

T = TypeVar("T")

TERMS = ("e_fis", "anchor", "sigreg_posa", "e_sem", "unif", "infonce", "sigreg_sem")
"""Terms of the objective, in the order of the tables."""

_DIRECTIONS = ("T2V", "V2T")
VALIDATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("val_loss", "loss_total"),
    *((f"val_{term}", f"loss_{term}") for term in TERMS),
    ("decision", "decision"),
    *((f"{d}_R@{k}", f"{d.lower()}_r{k}") for d in _DIRECTIONS for k in KS),
    *((f"{d}_P@{k}", f"{d.lower()}_precision{k}") for d in _DIRECTIONS for k in KS),
    *((f"{d}_Recall@{k}", f"{d.lower()}_recall{k}") for d in _DIRECTIONS for k in KS),
    *((f"{d}_MedR", f"{d.lower()}_medr") for d in _DIRECTIONS),
    *((f"{d}_MRR", f"{d.lower()}_mrr") for d in _DIRECTIONS),
    *((f"{d}_R@1_tolerant", f"tolerant_{d.lower()}_r1") for d in _DIRECTIONS),
    *((f"{d}_R@1_{end}", f"{d.lower()}_r1_{end}") for d in _DIRECTIONS for end in ("low", "high")),
    ("chance", "chance"),
    *((f"train_{d}_R@1", f"train_{d.lower()}_r1") for d in _DIRECTIONS),
    *((f"gap_{d}_R@1", f"gap_{d.lower()}_r1") for d in _DIRECTIONS),
    ("alignment", "alignment"),
    ("uniformity_video", "uniformity_video"),
    ("uniformity_text", "uniformity_text"),
    ("modality_gap", "modality_gap"),
    *((f"hubness_{d}", f"hubness_{d.lower()}") for d in _DIRECTIONS),
    *((f"noise_{d}_R@1", f"noise_{d.lower()}_r1") for d in _DIRECTIONS),
    ("noise_drop", "noise_drop"),
    *((f"{s}_{m}", f"{s}_{m}") for s in ("y", "text") for m in ("effective_rank", "isoscore")),
    ("y_sigreg", "y_sigreg"),
    ("text_sigreg", "text_sigreg"),
    ("r2_masked", "val_r2_masked"),
    ("r2_visible", "val_r2_visible"),
    ("gamma_masked", "val_gamma_masked"),
    ("keypoint_margin", "val_keypoint_margin"),
    ("dynamics_margin", "val_dynamics_margin"),
    ("leak_change", "leak_change"),
)
"""CSV column and the reading it holds."""
COLUMNS = (
    "epoch",
    "step",
    "reason",
    "stage",
    "lr",
    "train_loss",
    *(f"train_{term}" for term in TERMS),
    *(column for column, _ in VALIDATION_COLUMNS),
)
EPOCH_REASONS = frozenset({"epoch", "final"})
"""Validations that make a row of ``metrics.csv``; the others go to ``val_steps.csv``."""


def finish_time(eta: float | None) -> str:
    """The expected end of the run as ``dd/mm HH:MM``, or ``?`` without an estimate."""
    if eta is None:
        return "?"
    return f"{datetime.now() + timedelta(seconds=eta):%d/%m %H:%M}"


def _pair(readings: dict[str, float], key: str, digits: int = 4) -> str:
    values = [readings.get(f"{d.lower()}_{key}", math.nan) for d in _DIRECTIONS]
    return "/".join(f"{v:.{digits}f}" for v in values)


class RunReport:
    """Console lines, progress bars and CSV tables of one run (first GPU only)."""

    def __init__(self, directory: Path, enabled: bool) -> None:
        self.directory = directory
        self.enabled = enabled

    def bar(
        self, iterable: Iterable[T], desc: str, total: int | None = None, initial: int = 0
    ) -> Any:
        """A progress bar on standard error, as worldSign's: one refresh every 15 s at most."""
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            initial=initial,
            disable=not self.enabled,
            dynamic_ncols=True,
            mininterval=15.0,
            leave=False,
        )

    def say(self, line: str) -> None:
        if self.enabled:
            print(line, flush=True)

    def validation(  # noqa: PLR0913 (where the run is and what the validation read)
        self,
        *,
        reason: str,
        epoch: float,
        epochs: int,
        step: int,
        total_steps: int,
        stage: str,
        lr: float,
        train: dict[str, float],
        readings: dict[str, float],
        eta: float | None,
    ) -> None:
        """One line on standard output and one row in the run's tables."""
        if not self.enabled:
            return
        where = (
            f"epoch {epoch:g}/{epochs}"
            if reason in EPOCH_REASONS
            else f"val step {step}/{total_steps} ({reason})"
        )
        train_loss = train.get("loss", math.nan)
        self.say(
            f"[stage {stage}] {where} · step {step}/{total_steps} | "
            f"loss tr/val={train_loss:.4f}/{readings.get('loss_total', math.nan):.4f} | "
            f"R@1 T2V/V2T={_pair(readings, 'r1')} | R@5 T2V/V2T={_pair(readings, 'r5')} | "
            f"R@10 T2V/V2T={_pair(readings, 'r10')} | MedR T2V/V2T={_pair(readings, 'medr', 0)} | "
            f"MRR T2V/V2T={_pair(readings, 'mrr')} | "
            f"align={readings.get('alignment', math.nan):.4f} "
            f"unif v/t={readings.get('uniformity_video', math.nan):.3f}/"
            f"{readings.get('uniformity_text', math.nan):.3f} | lr={lr:.2e} | "
            f"end ≈ {finish_time(eta)}"
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "step": step,
            "reason": reason,
            "stage": stage,
            "lr": lr,
            "train_loss": train.get("loss", ""),
        }
        row |= {f"train_{term}": train.get(term, "") for term in TERMS}
        row |= {column: readings.get(key, "") for column, key in VALIDATION_COLUMNS}
        name = "metrics.csv" if reason in EPOCH_REASONS else "val_steps.csv"
        self._append(self.directory / name, row)

    def checkpoint(self, name: str, path: Path, step: int, epoch: float, note: str = "") -> None:
        self.say(f"[checkpoint] {name} -> {path} (step {step}, epoch {epoch:g}){note}")

    @staticmethod
    def _append(path: Path, row: dict[str, Any]) -> None:
        """A row matched to the file's own header by name: a resumed run that has gained a
        column never shifts the others (as worldSign's tables)."""
        header: Sequence[str] = COLUMNS
        fresh = not path.is_file() or path.stat().st_size == 0
        if not fresh:
            with path.open(newline="", encoding="utf-8") as stream:
                existing: list[str] = next(csv.reader(stream), [])
            header = existing or COLUMNS
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if fresh:
                writer.writerow(header)
            writer.writerow([row.get(column, "") for column in header])

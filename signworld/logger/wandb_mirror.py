"""An optional copy of the metrics log on Weights & Biases, offline by default.

Off unless ``diagnostics.wandb.enabled``; it needs the ``wandb`` package, which is not among
the project's dependencies: the environment that runs the training installs it if wanted.
Offline, the run is written under the run directory and uploaded later with ``wandb sync``
from a node with internet (the compute nodes of Leonardo and ReCaS have none), as worldSign
does. The W&B run's ID comes from the run directory, so every resume appends to the same run.
Every numeric reading of ``metrics.jsonl`` is logged as ``<kind>/<name>`` at its step.
"""

import hashlib
import importlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WandbMirror:
    def __init__(
        self,
        directory: Path,
        name: str,
        config: dict[str, Any],
        *,
        project: str,
        entity: str | None,
        mode: str,
    ) -> None:
        self.run: Any = None
        try:
            wandb: Any = importlib.import_module("wandb")
        except ImportError:
            logger.warning("diagnostics.wandb is on but wandb is not installed: not mirrored")
            return
        identity = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:16]
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            id=identity,
            resume="allow",
            mode=mode,
            dir=str(directory),
            config=config,
        )

    def write(self, kind: str, values: dict[str, Any]) -> None:
        if self.run is None:
            return
        numbers = {
            f"{kind}/{key}": value
            for key, value in values.items()
            if key != "step" and isinstance(value, int | float) and not isinstance(value, bool)
        }
        if numbers:
            step = values.get("step")
            self.run.log(numbers, step=step if isinstance(step, int) else None)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

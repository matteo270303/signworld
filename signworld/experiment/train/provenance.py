"""Reproducibility of a run: its seeds, and the record of what produced it.

* ``seed_everything`` runs before the model is built: Python, NumPy and torch draw the initial
  weights from the run's seed, so two launches of one configuration start from the same model.
* ``seed_step`` runs before every training step: the global generator, which the dropout of
  the predictors and of the text head draws from, restarts from the seed, the step and the
  GPU, as ``StepRandomness`` does for the masks, the SIGReg directions and the pose views. A
  resumed run draws what the uninterrupted one would have, with no generator state to save.
* ``Provenance`` keeps ``provenance.json`` in the run directory: one segment per launch or
  resume, with the code (commit, uncommitted changes saved as ``code-<n>.patch``, a hash of
  the source tree that exists even without git), the environment (Python, torch, CUDA, cuDNN,
  GPUs, host, job) and the hash of the configuration. A resume with another configuration is
  refused; one with other code is refused unless explicitly allowed, and the allowance is
  recorded. The inputs (weights, embeddings, index, statistics) are fingerprinted by P5.

GPU kernels are not bitwise deterministic (attention backward, atomic adds): the same seed
gives the same initialisation, data order, masks, directions, views and dropout, and runs that
agree up to floating-point reordering. The variance between seeds is measured apart (D4).
"""

import datetime
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from .config import WorldSignConfig

PROVENANCE: Final = "provenance.json"
STEP_STREAM: Final = 3
"""Key of the global generator's stream; ``StepRandomness`` uses 0, 1 and 2."""
SOURCE_PARTS: Final = ("signworld", "parameters", "pyproject.toml", "uv.lock")
"""What the source hash covers, relative to the repository root."""
REPOSITORY: Final = Path(__file__).resolve().parents[3]
JOB_VARIABLES: Final = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_NNODES",
    "SLURM_JOB_PARTITION",
    "CLUSTER_ID",
    "PROCESS_ID",
    "_CONDOR_JOB_AD",
)


def derive(seed: int, *key: int) -> int:
    """A 32-bit seed from ``seed`` and ``key``, as ``StepRandomness`` derives its own."""
    return int(np.random.SeedSequence([seed, *key]).generate_state(1)[0])


def seed_everything(seed: int) -> None:
    """Python, NumPy and torch (CPU and every GPU) from ``seed``: the initial weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_step(seed: int, step: int, rank: int) -> None:
    """The global generator of this GPU at ``step``: what dropout draws from."""
    torch.manual_seed(derive(seed, step, rank, STEP_STREAM))


def config_hash(config: WorldSignConfig) -> str:
    """SHA-256 of the configuration as the run sees it, keys in a fixed order."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def source_hash(root: Path) -> str:
    """SHA-256 over the source tree (``SOURCE_PARTS``): every file's path and contents."""
    digest = hashlib.sha256()
    files: list[Path] = []
    for part in SOURCE_PARTS:
        path = root / part
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout


@dataclass(frozen=True, slots=True)
class Code:
    source_sha256: str
    commit: str | None
    branch: str | None
    dirty: bool | None
    """True when tracked files differ from the commit; their diff is saved as ``patch``."""
    untracked: list[str]
    patch: str | None
    """File, in the run directory, with ``git diff HEAD`` at launch."""

    @classmethod
    def of(cls, root: Path) -> tuple["Code", str]:
        """The code at ``root`` and its uncommitted diff (empty without git or changes)."""
        commit = _git(root, "rev-parse", "HEAD")
        branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
        diff = _git(root, "diff", "HEAD") or ""
        untracked = _git(root, "ls-files", "--others", "--exclude-standard") or ""
        return (
            cls(
                source_sha256=source_hash(root),
                commit=commit.strip() if commit else None,
                branch=branch.strip() if branch else None,
                dirty=bool(diff) if commit else None,
                untracked=[line for line in untracked.splitlines() if line.endswith(".py")],
                patch=None,
            ),
            diff,
        )


@dataclass(frozen=True, slots=True)
class Environment:
    python: str
    torch: str
    cuda: str | None
    cudnn: int | None
    gpus: list[str]
    host: str
    platform: str
    job: dict[str, str]
    lock_sha256: str | None

    @classmethod
    def of(cls, root: Path) -> "Environment":
        lock = root / "uv.lock"
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return cls(
            python=sys.version.split()[0],
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
            gpus=[torch.cuda.get_device_name(i) for i in range(count)],
            host=socket.gethostname(),
            platform=platform.platform(),
            job={k: os.environ[k][:200] for k in JOB_VARIABLES if k in os.environ},
            lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
        )


@dataclass(frozen=True, slots=True)
class Segment:
    """One launch or resume of a run."""

    started: str
    command: list[str]
    resumed: bool
    world_size: int
    clips_per_gpu: int
    code: Code
    environment: Environment
    code_change_allowed: bool = False


class ProvenanceError(RuntimeError):
    """A resume that would mix two runs: another configuration, or other code."""


@dataclass
class Provenance:
    run: str
    config_sha256: str
    seed: int
    segments: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def read(cls, directory: Path) -> "Provenance | None":
        path = directory / PROVENANCE
        if not path.is_file():
            return None
        return cls(**json.loads(path.read_text()))

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f"{PROVENANCE}.tmp"
        temporary.write_text(json.dumps(asdict(self), indent=2) + "\n")
        temporary.replace(directory / PROVENANCE)


def record_launch(  # noqa: PLR0913 (what a launch is made of)
    directory: Path,
    config: WorldSignConfig,
    *,
    root: Path = REPOSITORY,
    resumed: bool,
    world_size: int,
    clips_per_gpu: int,
    allow_code_change: bool = False,
) -> Provenance:
    """Check a resume against the run's first launch, then append this launch's segment.

    Call it on the first GPU only. ``root`` is the repository the code runs from.
    """
    if resumed and Provenance.read(directory) is None:
        raise ProvenanceError(f"{directory}: a checkpoint without provenance.json")
    digest = config_hash(config)
    code, diff = Code.of(root)
    found = Provenance.read(directory)
    if found is None:
        provenance = Provenance(config.name, digest, config.training.seed)
    else:
        if found.config_sha256 != digest:
            raise ProvenanceError(
                f"{directory}: the configuration differs from the run's ({digest[:12]} against "
                f"{found.config_sha256[:12]}); start a new run directory"
            )
        first = found.segments[0]["code"]["source_sha256"] if found.segments else None
        if first not in (None, code.source_sha256) and not allow_code_change:
            raise ProvenanceError(
                f"{directory}: the code differs from the run's first launch; resume with "
                "--allow-code-change to accept it (it is recorded)"
            )
        provenance = found
    number = len(provenance.segments)
    if diff:
        patch = f"code-{number}.patch"
        (directory / patch).parent.mkdir(parents=True, exist_ok=True)
        (directory / patch).write_text(diff)
        code = Code(code.source_sha256, code.commit, code.branch, code.dirty, code.untracked, patch)
    segment = Segment(
        started=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        command=list(sys.argv),
        resumed=resumed,
        world_size=world_size,
        clips_per_gpu=clips_per_gpu,
        code=code,
        environment=Environment.of(root),
        code_change_allowed=allow_code_change and number > 0,
    )
    provenance.segments.append(asdict(segment))
    provenance.write(directory)
    return provenance

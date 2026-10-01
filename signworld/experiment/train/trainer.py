"""The training loop of a WorldSign run (§4.10).

Every step: the stage of the curriculum decides the passes and what trains; the learning
rate follows warm-up, constant phase and cooldown; the model runs under bf16 autocast inside
DDP; AdamW steps; a non-finite loss on any GPU stops the run (§4.13.3).

The constant phase ends in one of two ways: early stopping (no gain of the decision metric on
the held-out channel split for ``patience`` epochs) or the planned number of steps minus the
cooldown. Either way the run goes back to the best checkpoint and cools down from there
(V-JEPA 2: several cooldowns can start from checkpoints of the constant phase). Validation and
a checkpoint come every ``validation_every`` steps and at every epoch end; a run killed at any
point resumes from its last checkpoint, at the same place in the sampler's order. With
``diagnostics.cadence = epoch`` every diagnostic runs at the end of each epoch instead, and
``validation_every`` only spaces the checkpoints. Every
``latest`` checkpoint also writes the energies of every training clip since the previous one
(``checkpoints/energies``), for the audit of the high-energy tail (§4.13.4). What the run
prints and tabulates (progress bars, a line per validation and checkpoint, ``metrics.csv``)
is ``reporting``'s, set up as in the worldSign runs.
"""

import dataclasses
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .checkpoint import CheckpointStore, TrainingState
from .config import WorldSignConfig
from .curriculum import (
    CosineSchedule,
    Curriculum,
    LearningRateSchedule,
    Stage,
    families,
    trainable_names,
)
from .data import ClipDataset, Collate, EpochSampler
from .distributed import SINGLE, Distributed
from .model import StepRandomness, WorldSign, WorldSignBatch
from .monitor import Monitor, RunStoppedError
from .reporting import RunReport, finish_time
from .validation import RetrievalScores


class NonFiniteLossError(RuntimeError):
    """A NaN or Inf in the loss of some GPU: the run stops (§4.13.3)."""


@dataclasses.dataclass
class Progress:
    """Wall-clock bookkeeping of the progress line: steps timed apart from evaluations."""

    step: int | None = None
    """The step of the last reading; None before the first."""
    time: float = 0.0
    paused: float = 0.0
    """Seconds spent in validations and rare readings since the last reading."""
    per_step: float | None = None
    """Moving average of the wall-clock seconds of one step, data loading included."""
    validations: list[float] = dataclasses.field(default_factory=list)
    rares: list[float] = dataclasses.field(default_factory=list)

    def advance(self, step: int, now: float) -> None:
        if self.step is not None and step > self.step:
            measured = max(now - self.time - self.paused, 0.0) / (step - self.step)
            self.per_step = (
                measured if self.per_step is None else 0.7 * self.per_step + 0.3 * measured
            )
        self.step, self.time, self.paused = step, now, 0.0

    def eta(self, steps: int, validations: int, rares: int) -> float | None:
        """Seconds left: the steps at the current pace, the evaluations at their mean length."""
        if self.per_step is None:
            return None

        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        return (
            steps * self.per_step + validations * mean(self.validations) + rares * mean(self.rares)
        )


class MetricsLog:
    """One JSON line per record, written by the first GPU only."""

    def __init__(self, path: Path, collective: Distributed = SINGLE) -> None:
        self.path = path
        self.enabled = collective.is_main
        if self.enabled:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **values: Any) -> None:
        if not self.enabled:
            return
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"kind": kind, "time": time.time(), **values}) + "\n")


class Trainer:
    def __init__(  # noqa: PLR0913, PLR0915, PLR0917 (the model, its data, where it runs and writes)
        self,
        model: WorldSign,
        config: WorldSignConfig,
        train_data: ClipDataset,
        validation_data: ClipDataset,
        output: Path,
        collective: Distributed = SINGLE,
        probe_data: ClipDataset | None = None,
        train_subset_data: ClipDataset | None = None,
    ) -> None:
        """``train_subset_data``: training clips as many as the validation clips, for R@1 on
        training data and its gap to validation."""
        training = config.training
        if training.batch_size % collective.world_size:
            raise ValueError(
                f"batch {training.batch_size} does not split over {collective.world_size} GPUs"
            )
        self.config = config
        self.collective = collective
        self.device = collective.device
        self.model = model.to(self.device)
        self.trainable = families(model)
        names = trainable_names(model)
        self.bf16 = training.precision == "bf16"

        groups = model.parameter_groups(training)
        for group in groups:
            group["base_lr"] = group["lr"]
        self.optimizer = torch.optim.AdamW(
            groups, betas=training.betas, fused=self.device.type == "cuda"
        )
        self.per_gpu = training.batch_size // collective.world_size
        collate = Collate(model.text.centering.languages)
        self.sampler = EpochSampler(
            len(train_data), collective.rank, collective.world_size, training.seed
        )
        self.loader = self._loader(train_data, self.sampler, collate)
        validation_sampler = EpochSampler(
            len(validation_data), collective.rank, collective.world_size, 0, shuffle=False
        )
        # Every validation clip counts: the last, smaller batch is kept (the GPUs still get
        # equal shares, and nothing collective runs per batch during validation).
        self.validation_loader = self._loader(
            validation_data, validation_sampler, collate, drop_last=False
        )
        self.train_subset_loader = None
        if train_subset_data is not None:
            subset_sampler = EpochSampler(
                len(train_subset_data), collective.rank, collective.world_size, 0, shuffle=False
            )
            self.train_subset_loader = self._loader(
                train_subset_data, subset_sampler, collate, drop_last=False
            )

        self.steps_per_epoch = self.sampler.per_rank // self.per_gpu
        if self.steps_per_epoch == 0:
            raise ValueError("fewer training clips than one batch")
        self.total_steps = training.epochs * self.steps_per_epoch
        cooldown = max(1, round(training.cooldown_fraction * self.total_steps))
        self.schedule = LearningRateSchedule(
            max(1, round(training.warmup_fraction * self.total_steps)), cooldown
        )
        self.constant_end = self.total_steps - cooldown
        pose = config.pose_encoder
        self.final_layer_schedule = CosineSchedule(
            max(1, round(pose.final_layer_warmup * self.total_steps)),
            round(pose.final_layer_decay_end * self.total_steps),
        )
        self.curriculum = Curriculum(training.stages, self.total_steps, model.has_physical_level)
        self.checkpoints = CheckpointStore(output / "checkpoints", names, collective)
        self.log = MetricsLog(output / "metrics.jsonl", collective)
        self.output = output
        self.monitor = Monitor(
            model,
            config,
            self._probe(probe_data, collate),
            self.log,
            collective,
            total_steps=self.total_steps,
        )
        self.stops = self._stop_steps()
        self.stage: Stage | None = None
        self.wrapped: nn.Module = model
        self.energies: list[dict[str, list[Any]]] = []
        """This GPU's energies of every training clip since the last ``latest`` checkpoint."""
        self.at_epoch_end = config.diagnostics.cadence == "epoch"
        """Every diagnostic at the end of each epoch only (frequent, validation, rare, stops)."""
        self.progress = Progress()
        self.report = RunReport(output, collective.is_main)
        self.epoch_sums: dict[str, float] = defaultdict(float)
        """This GPU's sums of the loss and its terms over the steps of the current epoch."""
        self.epoch_steps = 0
        self.latest: tuple[int, RetrievalScores] | None = None
        """The step and the scores of the latest validation, for the progress line."""

    def _probe(self, data: ClipDataset | None, collate: Collate) -> list[WorldSignBatch]:
        """This GPU's share of the fixed probe clips, in small batches, kept on the CPU."""
        if data is None:
            return []
        count = min(len(data), self.config.diagnostics.probe_clips)
        mine = list(range(self.collective.rank, count, self.collective.world_size))
        size = max(1, min(self.per_gpu, 32))
        return [collate([data[i] for i in mine[s : s + size]]) for s in range(0, len(mine), size)]

    def _stop_steps(self) -> dict[int, list[str]]:
        """Where the programmed stops of §4.13.5 fall: F1 at the end of stage 1, F2 at 10 % of
        stage 2, F3 at 30 % of the run."""
        bounds = {stage.name: start for start, stage in self.curriculum.bounds}
        stops: dict[int, list[str]] = {}
        if self.model.has_physical_level and "2a" in bounds:
            stops.setdefault(max(1, bounds["2a"]), []).append("F1")
        start = bounds["2"]
        second = start + max(1, round(0.1 * max(self.constant_end - start, 1)))
        stops.setdefault(second, []).append("F2")
        stops.setdefault(max(1, round(0.3 * self.total_steps)), []).append("F3")
        return stops

    def _loader(
        self, data: ClipDataset, sampler: EpochSampler, collate: Collate, *, drop_last: bool = True
    ) -> DataLoader[Any]:
        workers = self.config.data.workers
        return DataLoader(
            data,
            batch_size=self.per_gpu,
            sampler=sampler,
            collate_fn=collate,
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            drop_last=drop_last,
            persistent_workers=workers > 0,
            prefetch_factor=self.config.data.prefetch if workers > 0 else None,
        )

    # ------------------------------------------------------------------ the run

    def fit(self) -> TrainingState:
        state = self._resume()
        try:
            return self._fit(state)
        except RunStoppedError:
            self._save("stopped", state)
            raise

    def _fit(self, state: TrainingState) -> TrainingState:
        if state.step == 0:  # everything read once at step 0, as the reference (§4.13)
            self._validate(state, "step 0")
            self._rare(state)
        self.report.say(
            f"[run] {self.total_steps} steps ({self.steps_per_epoch} per epoch, "
            f"{self.config.training.epochs} epochs), constant phase to {self.constant_end}, "
            f"cooldown {self.schedule.cooldown_steps} steps"
        )
        self.progress.advance(state.step, time.perf_counter())
        while not state.finished:
            self._run_epoch(state)
        scores = self._validate(state, "final")
        if self.at_epoch_end:
            self._rare(state)
        self.checkpoints.save("final", self.model, self.optimizer, state)
        self.report.checkpoint(
            "final", self.checkpoints.path("final"), state.step, self._epochs_done(state)
        )
        self.log.write("summary", step=state.step, best_step=state.best_step, **scores.as_log())
        return state

    def _resume(self) -> TrainingState:
        if self.checkpoints.exists("latest"):
            state = self.checkpoints.load("latest", self.model, self.optimizer)
            saved = self._monitor_file()
            if saved.is_file():
                self.monitor.load(torch.load(saved, map_location="cpu", weights_only=False))
            self.report.say(f"[run] resuming at step {state.step} (epoch {state.epoch + 1})")
            return state
        return TrainingState()

    def _monitor_file(self) -> Path:
        return self.output / "checkpoints" / f"monitor.rank{self.collective.rank}.pt"

    def _save(self, name: str, state: TrainingState) -> None:
        """A checkpoint, with the monitor's references and history next to it."""
        self.checkpoints.save(name, self.model, self.optimizer, state)
        self.report.checkpoint(
            name, self.checkpoints.path(name), state.step, self._epochs_done(state)
        )
        if name == "latest":
            torch.save(self.monitor.state(), self._monitor_file())
            self._write_energies(state)

    def _keep_energies(self, batch: WorldSignBatch, step: int, samples: dict[str, Any]) -> None:
        if batch.clip_ids is None or not samples:
            return
        count = len(batch.clip_ids)
        row: dict[str, list[Any]] = {"clip_id": list(batch.clip_ids), "step": [step] * count}
        for name in ("e_sem", "e_fis"):
            value = samples.get(name)
            row[name] = value.float().tolist() if value is not None else [None] * count
        self.energies.append(row)

    def _write_energies(self, state: TrainingState) -> None:
        """One file per GPU: clip ID, step, E_sem and E_fis (null when the pass did not run)."""
        if not self.energies:
            return
        schema = pa.schema(
            [
                ("clip_id", pa.string()),
                ("step", pa.int64()),
                ("e_sem", pa.float32()),
                ("e_fis", pa.float32()),
            ]
        )
        columns = {name: [v for row in self.energies for v in row[name]] for name in schema.names}
        path = (
            self.output
            / "checkpoints"
            / "energies"
            / f"step{state.step:08d}.rank{self.collective.rank}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns, schema=schema), path)
        self.energies.clear()

    def _run_epoch(self, state: TrainingState) -> None:
        self.sampler.configure(state.epoch, state.position)
        self.epoch_sums, self.epoch_steps = defaultdict(float), 0
        bar = self.report.bar(
            self.loader,
            f"[train] epoch {state.epoch + 1}/{self.config.training.epochs}",
            total=self.steps_per_epoch,
            initial=state.position // self.per_gpu,
        )
        try:
            for batch in bar:
                self._step(batch, state)
                if self.report.enabled:
                    bar.set_postfix_str(self._postfix(state), refresh=False)
                if self._after_step(state):
                    return
        finally:
            bar.close()
        state.epoch += 1
        state.position = 0
        if state.cooldown_start is None:
            self._end_of_epoch(state)

    def _after_step(self, state: TrainingState) -> bool:
        """Validation, checkpoints and phase changes; True when the order of clips changes."""
        if state.step % self.config.training.validation_every == 0:
            if not self.at_epoch_end:
                self._validate(state, "periodic")
            self._save("latest", state)
        if not self.at_epoch_end and state.step % self.config.diagnostics.rare_every == 0:
            self._rare(state)
        stops = self.stops.get(state.step, [])
        for name in stops if state.cooldown_start is None and not self.at_epoch_end else []:
            self._stop_point(name, state)
        if state.cooldown_start is None and state.step >= self.constant_end:
            self._consider_best(state, self._validate(state, "end of constant phase"))
            self._start_cooldown(state, "planned steps")
            return True
        end = (
            None
            if state.cooldown_start is None
            else state.cooldown_start + self.schedule.cooldown_steps
        )
        if end is not None and state.step >= end:
            state.finished = True
            return True
        return False

    def _stop_point(self, name: str, state: TrainingState, *, fresh: bool = True) -> None:
        """A programmed stop: fresh readings, the criteria, and in the gate run the stop.

        ``fresh=False``: judged on the readings just taken at the end of the epoch.
        """
        if fresh:
            self._validate(state, name)
        extrapolated = None
        if name == "F3":
            if fresh:
                self._rare(state)
            extrapolated = self.monitor.extrapolate()
        report = self.monitor.stop_point(name, state.step, extrapolated)
        failed = [label for label, (ok, _) in report.criteria.items() if not ok]
        self.report.say(
            f"[stop] {name} at step {state.step}: "
            + ("passed" if report.passed else f"FAILED ({'; '.join(failed)})")
        )
        if not report.passed and self.config.diagnostics.gate_stops:
            self._save(f"stop_{name}", state)
            raise RunStoppedError(f"stop {name} failed at step {state.step}: {failed}")

    def _end_of_epoch(self, state: TrainingState) -> None:
        improved = self._consider_best(state, self._validate(state, "epoch"))
        if self.at_epoch_end:
            self._rare(state)
            for at, names in sorted(self.stops.items()):
                if state.step - self.steps_per_epoch < at <= state.step:
                    for name in names:
                        self._stop_point(name, state, fresh=False)
        if not improved:
            state.bad_epochs += 1
        self._save("latest", state)
        if state.bad_epochs >= self.config.training.patience:
            self._start_cooldown(state, "early stopping")

    def _consider_best(self, state: TrainingState, scores: RetrievalScores) -> bool:
        if scores.decision <= state.best_metric:
            return False
        state.best_metric, state.best_step, state.bad_epochs = scores.decision, state.step, 0
        self.checkpoints.save("best", self.model, self.optimizer, state)
        self.report.checkpoint(
            "best",
            self.checkpoints.path("best"),
            state.step,
            self._epochs_done(state),
            f" · NEW BEST decision {scores.decision:.4f}",
        )
        return True

    def _start_cooldown(self, state: TrainingState, reason: str) -> None:
        """Back to the best checkpoint; the cooldown starts from its step."""
        if self.checkpoints.exists("best"):
            restored = self.checkpoints.load("best", self.model, self.optimizer)
            for field in ("step", "epoch", "position", "best_metric", "best_step"):
                setattr(state, field, getattr(restored, field))
        state.cooldown_start = state.step
        state.bad_epochs = 0
        self.log.write("cooldown", step=state.step, reason=reason)
        self.report.say(f"[cooldown] from step {state.step} ({reason})")
        self._save("latest", state)

    # ------------------------------------------------------------------ one step

    def _enter(self, stage: Stage, step: int) -> None:
        """Freeze and unfreeze for the stage; DDP must be rebuilt over the new trainable set."""
        Curriculum.apply(stage, self.trainable)
        self.stage = stage
        self.wrapped = self.model
        if self.collective.active:
            self.wrapped = DistributedDataParallel(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                find_unused_parameters=self.config.training.find_unused_parameters,
            )
        self.report.say(f"[stage] {stage.name} from step {step}")

    def _step(self, batch: WorldSignBatch, state: TrainingState) -> None:
        cooling = state.cooldown_start is not None
        stage = self.curriculum.last if cooling else self.curriculum.stage_at(state.step)
        if stage != self.stage:
            self._enter(stage, state.step)
            self.log.write("stage", step=state.step, stage=stage.name)
        start = self.constant_end if state.cooldown_start is None else state.cooldown_start
        factor = self.schedule.factor(state.step, start)
        final_factor = self.final_layer_schedule.factor(state.step)
        for group in self.optimizer.param_groups:
            own = final_factor if group["schedule"] == "final_layer" else factor
            group["lr"] = group["base_lr"] * own
        began = time.perf_counter()
        batch = batch.to(self.device).augmented()
        frequent = self._frequent_due(state)
        if frequent and state.step == 0:
            self.monitor.frequent(0, batch, stage)  # the untrained model: the reference
        randomness = StepRandomness.at(self.config.training.seed, state.step, self.collective.rank)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            terms = self.wrapped(
                batch,
                state.step,
                self.total_steps,
                randomness,
                physical=stage.physical,
                semantic=stage.semantic,
            )
        terms.total.backward()
        clip = self.config.training.gradient_clip
        if clip is not None:
            nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], clip)
        gradients = self.monitor.lora_gradients() if frequent else {}
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        total = float(terms.total.detach())
        if self.collective.any(not math.isfinite(total)):
            raise NonFiniteLossError(f"non-finite loss at step {state.step}: {total}")
        if frequent and state.step > 0:
            self.monitor.frequent(state.step, batch, stage, gradients)
        parts = {name: float(value.detach()) for name, value in terms.parts.items()}
        self.monitor.after_step(state.step, total, parts)
        self.epoch_steps += 1
        for name, value in ({"loss": total} | parts).items():
            self.epoch_sums[name] += value
        self._keep_energies(batch, state.step, terms.samples)
        state.step += 1
        state.position += self.per_gpu
        if state.step % self.config.training.log_every == 0:
            self._log_step(state, stage, total, terms.parts | terms.diagnostics, factor, began)

    def _log_step(
        self,
        state: TrainingState,
        stage: Stage,
        total: float,
        parts: dict[str, torch.Tensor],
        factor: float,
        began: float,
    ) -> None:
        mean: Callable[[float], float] = self.collective.mean
        seconds = time.perf_counter() - began
        loss = mean(total)
        values = {name: mean(float(value.detach())) for name, value in parts.items()}
        memory = (
            torch.cuda.max_memory_allocated(self.device) / 2**30
            if self.device.type == "cuda"
            else 0.0
        )
        self.progress.advance(state.step, time.perf_counter())
        eta = self._eta(state)
        self.log.write(
            "step",
            step=state.step,
            epoch=state.epoch,
            stage=stage.name,
            lr_factor=factor,
            seconds=seconds,
            clips_per_second=self.config.training.batch_size / max(seconds, 1e-9),
            memory_gib=memory,
            loss=loss,
            eta_seconds=eta,
            **values,
        )

    def _frequent_due(self, state: TrainingState) -> bool:
        """Step 0, the reference; then every ``frequent_every`` steps, or with the diagnostics
        at epoch end the last step of every epoch."""
        if not self.at_epoch_end:
            return self.monitor.due(state.step, self.config.diagnostics.frequent_every)
        last = state.position // self.per_gpu + 1 == self.steps_per_epoch
        return state.step == 0 or last

    def _end_step(self, state: TrainingState) -> int:
        """The last step: the planned one, or the end of the cooldown once it has begun."""
        if state.cooldown_start is None:
            return self.total_steps
        return state.cooldown_start + self.schedule.cooldown_steps

    def _evaluations_left(self, state: TrainingState) -> tuple[int, int]:
        """Validations and rare readings still to come, if the run goes to its planned end."""
        training, diagnostics = self.config.training, self.config.diagnostics
        step, end = state.step, self._end_step(state)
        epoch_ends = sum(
            step < epoch * self.steps_per_epoch <= self.constant_end
            for epoch in range(1, training.epochs + 1)
        )
        if self.at_epoch_end:
            # The final readings; before the cooldown, the epoch ends and the end of the
            # constant phase (validated only: it chooses where the cooldown starts).
            cooling = state.cooldown_start is not None
            validations = 1 if cooling else 2 + epoch_ends
            rares = 1 if cooling else 1 + epoch_ends
            return validations, rares
        validations = end // training.validation_every - step // training.validation_every + 1
        rares = end // diagnostics.rare_every - step // diagnostics.rare_every
        if state.cooldown_start is None:
            validations += 1 + epoch_ends  # the end of the constant phase and the epoch ends
            for at, names in self.stops.items():
                if at > step:
                    validations += len(names)
                    rares += names.count("F3")
        return validations, rares

    def _eta(self, state: TrainingState) -> float | None:
        return self.progress.eta(self._end_step(state) - state.step, *self._evaluations_left(state))

    def _epochs_done(self, state: TrainingState) -> float:
        return round(state.step / self.steps_per_epoch, 2)

    def _postfix(self, state: TrainingState) -> str:
        """The bar's tail: the stage, the epoch's running loss, the latest R@1, the end."""
        loss = self.epoch_sums.get("loss", math.nan) / max(self.epoch_steps, 1)
        recall = "?"
        if self.latest is not None:
            at, scores = self.latest
            recall = f"{scores.t2v[1]:.4f}/{scores.v2t[1]:.4f} (step {at})"
        stage = self.stage.name if self.stage is not None else "?"
        return (
            f"stage {stage} · loss {loss:.4f} · R@1 T2V/V2T {recall} · "
            f"end ≈ {finish_time(self._eta(state))}"
        )

    def _train_means(self) -> dict[str, float]:
        """The current epoch's mean loss and terms, over every GPU: every GPU must call it."""
        steps = max(self.epoch_steps, 1)
        return {
            name: self.collective.mean(total / steps) for name, total in self.epoch_sums.items()
        }

    def _rare(self, state: TrainingState) -> None:
        began = time.perf_counter()
        self.monitor.rare(state.step, self.validation_loader)
        self._paused(self.progress.rares, began)

    def _paused(self, durations: list[float], began: float) -> None:
        """Time spent in an evaluation: kept for the ETA, left out of the time per step."""
        seconds = time.perf_counter() - began
        durations.append(seconds)
        self.progress.paused += seconds

    def _validate(self, state: TrainingState, reason: str) -> RetrievalScores:
        began = time.perf_counter()
        languages = self.model.text.centering.languages
        scores = self.monitor.validation(
            state.step, self.validation_loader, languages, self.train_subset_loader
        )
        self._paused(self.progress.validations, began)
        self.latest = (state.step, scores)
        self.log.write(
            "checkpoint_validation", step=state.step, reason=reason, decision=scores.decision
        )
        self.report.validation(
            reason=reason,
            epoch=self._epochs_done(state),
            epochs=self.config.training.epochs,
            step=state.step,
            total_steps=self._end_step(state),
            stage=self.stage.name if self.stage is not None else "-",
            lr=self.optimizer.param_groups[0]["lr"],
            train=self._train_means(),
            readings=self.monitor.last_validation,
            eta=self._eta(state),
        )
        return scores

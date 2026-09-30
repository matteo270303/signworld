"""The training loop of a WorldSign run (§4.10).

Every step: the stage of the curriculum decides the passes and what trains; the learning
rate follows warm-up, constant phase and cooldown; the model runs under bf16 autocast inside
DDP; AdamW steps; a non-finite loss on any GPU stops the run (§4.13.3).

The constant phase ends in one of two ways: early stopping (no gain of the decision metric on
the held-out channel split for ``patience`` epochs) or the planned number of steps minus the
cooldown. Either way the run goes back to the best checkpoint and cools down from there
(V-JEPA 2: several cooldowns can start from checkpoints of the constant phase). Validation and
a checkpoint come every ``validation_every`` steps and at every epoch end; a run killed at any
point resumes from its last checkpoint, at the same place in the sampler's order.
"""

import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .checkpoint import CheckpointStore, TrainingState
from .config import WorldSignConfig
from .curriculum import CosineSchedule, Curriculum, LearningRateSchedule, Stage, families
from .data import ClipDataset, Collate, EpochSampler
from .distributed import SINGLE, Distributed
from .model import StepRandomness, WorldSign, WorldSignBatch
from .validation import RetrievalScores, validate

logger = logging.getLogger(__name__)


class NonFiniteLossError(RuntimeError):
    """A NaN or Inf in the loss of some GPU: the run stops (§4.13.3)."""


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
    def __init__(
        self,
        model: WorldSign,
        config: WorldSignConfig,
        train_data: ClipDataset,
        validation_data: ClipDataset,
        output: Path,
        collective: Distributed = SINGLE,
    ) -> None:
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
        names = {id(p): n for n, p in model.named_parameters()}
        trainable_names = {names[id(p)] for group in self.trainable.values() for p in group}
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
        self.validation_loader = self._loader(validation_data, validation_sampler, collate)

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
        self.checkpoints = CheckpointStore(output / "checkpoints", trainable_names, collective)
        self.log = MetricsLog(output / "metrics.jsonl", collective)
        self.stage: Stage | None = None
        self.wrapped: nn.Module = model

    def _loader(
        self, data: ClipDataset, sampler: EpochSampler, collate: Collate
    ) -> DataLoader[Any]:
        workers = self.config.data.workers
        return DataLoader(
            data,
            batch_size=self.per_gpu,
            sampler=sampler,
            collate_fn=collate,
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
            persistent_workers=workers > 0,
            prefetch_factor=self.config.data.prefetch if workers > 0 else None,
        )

    # ------------------------------------------------------------------ the run

    def fit(self) -> TrainingState:
        state = self._resume()
        logger.info(
            "%d steps (%d per epoch), constant phase to %d, cooldown %d steps",
            self.total_steps,
            self.steps_per_epoch,
            self.constant_end,
            self.schedule.cooldown_steps,
        )
        while not state.finished:
            self._run_epoch(state)
        scores = self._validate(state, "final")
        self.checkpoints.save("final", self.model, self.optimizer, state)
        self.log.write("summary", step=state.step, best_step=state.best_step, **scores.as_log())
        return state

    def _resume(self) -> TrainingState:
        if self.checkpoints.exists("latest"):
            state = self.checkpoints.load("latest", self.model, self.optimizer)
            logger.info("Resuming at step %d (epoch %d)", state.step, state.epoch)
            return state
        return TrainingState()

    def _run_epoch(self, state: TrainingState) -> None:
        self.sampler.configure(state.epoch, state.position)
        for batch in self.loader:
            self._step(batch, state)
            if self._after_step(state):
                return
        state.epoch += 1
        state.position = 0
        if state.cooldown_start is None:
            self._end_of_epoch(state)

    def _after_step(self, state: TrainingState) -> bool:
        """Validation, checkpoints and phase changes; True when the order of clips changes."""
        if state.step % self.config.training.validation_every == 0:
            self._validate(state, "periodic")
            self.checkpoints.save("latest", self.model, self.optimizer, state)
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

    def _end_of_epoch(self, state: TrainingState) -> None:
        improved = self._consider_best(state, self._validate(state, "epoch"))
        if not improved:
            state.bad_epochs += 1
        self.checkpoints.save("latest", self.model, self.optimizer, state)
        if state.bad_epochs >= self.config.training.patience:
            self._start_cooldown(state, "early stopping")

    def _consider_best(self, state: TrainingState, scores: RetrievalScores) -> bool:
        if scores.decision <= state.best_metric:
            return False
        state.best_metric, state.best_step, state.bad_epochs = scores.decision, state.step, 0
        self.checkpoints.save("best", self.model, self.optimizer, state)
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
        logger.info("Cooldown from step %d (%s)", state.step, reason)
        self.checkpoints.save("latest", self.model, self.optimizer, state)

    # ------------------------------------------------------------------ one step

    def _enter(self, stage: Stage) -> None:
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
        logger.info("Stage %s from step", stage.name)

    def _step(self, batch: WorldSignBatch, state: TrainingState) -> None:
        cooling = state.cooldown_start is not None
        stage = self.curriculum.last if cooling else self.curriculum.stage_at(state.step)
        if stage != self.stage:
            self._enter(stage)
            self.log.write("stage", step=state.step, stage=stage.name)
        start = self.constant_end if state.cooldown_start is None else state.cooldown_start
        factor = self.schedule.factor(state.step, start)
        final_factor = self.final_layer_schedule.factor(state.step)
        for group in self.optimizer.param_groups:
            own = final_factor if group["schedule"] == "final_layer" else factor
            group["lr"] = group["base_lr"] * own
        began = time.perf_counter()
        batch = batch.to(self.device)
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
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        total = float(terms.total.detach())
        if self.collective.any(not math.isfinite(total)):
            raise NonFiniteLossError(f"non-finite loss at step {state.step}: {total}")
        state.step += 1
        state.position += self.per_gpu
        if state.step % self.config.training.log_every == 0:
            self._log_step(state, stage, terms.parts | terms.diagnostics, factor, began)

    def _log_step(
        self,
        state: TrainingState,
        stage: Stage,
        parts: dict[str, torch.Tensor],
        factor: float,
        began: float,
    ) -> None:
        mean: Callable[[float], float] = self.collective.mean
        seconds = time.perf_counter() - began
        values = {name: mean(float(value.detach())) for name, value in parts.items()}
        memory = (
            torch.cuda.max_memory_allocated(self.device) / 2**30
            if self.device.type == "cuda"
            else 0.0
        )
        self.log.write(
            "step",
            step=state.step,
            epoch=state.epoch,
            stage=stage.name,
            lr_factor=factor,
            seconds=seconds,
            clips_per_second=self.config.training.batch_size / max(seconds, 1e-9),
            memory_gib=memory,
            **values,
        )

    def _validate(self, state: TrainingState, reason: str) -> RetrievalScores:
        scores = validate(
            self.model, self.validation_loader, self.device, self.collective, bf16=self.bf16
        )
        self.log.write("validation", step=state.step, reason=reason, **scores.as_log())
        logger.info(
            "Step %d, %s: R@1 T2V %.4f, V2T %.4f",
            state.step,
            reason,
            scores.t2v[1],
            scores.v2t[1],
        )
        return scores

"""CUDA kernels of one training step of a stage, the run's model and configuration.

``signworld train profile`` says which section takes the time; this lists the kernels, for
the backward above all (activation checkpointing, LoRA, attention).

    PYTHONPATH=. .venv/bin/python scripts/perf/trace_step.py -c ... --output tmp/x --stage F --clips 4
"""

import argparse
from pathlib import Path

import torch

import signworld  # noqa: F401  (thread caps)
from signworld.experiment.train.checkpoint import TrainingState
from signworld.experiment.train.config import WorldSignConfig, load_config, merge
from signworld.experiment.train.curriculum import STAGES, Curriculum
from signworld.experiment.train.distributed import Distributed
from signworld.experiment.train.run import prepare
from signworld.models.worldsign.model import StepRandomness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", default="F")
    parser.add_argument("--clips", type=int, default=4)
    parser.add_argument("--rows", type=int, default=30)
    args = parser.parse_args()
    settings = load_config(*args.config)
    batch_change = {"training": {"batch_size": args.clips, "effective_batch": args.clips}}
    settings = WorldSignConfig.model_validate(merge(settings.model_dump(mode="json"), batch_change))
    collective = Distributed.from_environment()
    trainer = prepare(settings, args.output, collective).trainer
    batch = next(iter(trainer.loader))
    stage = STAGES[args.stage]
    Curriculum.apply(stage, trainer.trainable)
    state = TrainingState()
    state.step = trainer.curriculum.last_start + trainer.steps_per_epoch
    model, device = trainer.model, trainer.device

    def step() -> None:
        trainer._set_rates(state)
        moved = batch.to(device).augmented()
        randomness = StepRandomness.at(0, state.step, 0)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            terms = model.loss(moved, state.step, trainer.total_steps, randomness,
                               pose=stage.pose, physical=stage.physical, semantic=stage.semantic)  # fmt: skip
        terms.total.backward()  # type: ignore[no-untyped-call]
        trainer.optimizer.step()
        trainer.optimizer.zero_grad(set_to_none=True)

    for _ in range(3):  # compilation
        step()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.rows, max_name_column_width=90))
    collective.shutdown()


if __name__ == "__main__":
    main()

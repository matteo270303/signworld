"""Stato dei run e lettura dei job di debug. Uso (dalla radice del repo):

    python slurm/run_report.py status    ultimo passo di ogni run in runs/
    python slurm/run_report.py debug     memoria e secondi per stage dei run in runs/debug/

Il debug confronta il picco di memoria di ogni GPU (gpu.csv, nvidia-smi) con la sua capienza e
stima la durata del run vero dai secondi per passo di ogni stage (solo i passi: validazioni e
diagnostiche escluse).
"""

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

RUNS = Path("runs")
SAFE_SHARE = 0.90  # oltre il 90 % della memoria della GPU: rischio di OOM
REAL = [  # il run vero di riferimento, braccio A (slurm/experiments.sh)
    "parameters/model/worldsign.yaml",
    "parameters/model/youtube_sl25.yaml",
    "parameters/model/youtube_sl25_run.yaml",
    "parameters/ablation/arm_A.yaml",
]
OOM = re.compile(r"OutOfMemoryError|CUDA out of memory")
COLUMNS = 4  # timestamp, GPU, memoria usata, memoria totale (+ utilizzo)


def records(run: Path) -> list[dict]:
    path = run / "metrics.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def gpu_peak(run: Path) -> tuple[float, float] | None:
    """Picco di memoria usata su tutte le GPU e capienza, in GiB, da gpu.csv."""
    path = run / "gpu.csv"
    if not path.is_file():
        return None
    used, total = 0.0, 0.0
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < COLUMNS or not parts[2].split()[0].isdigit():
            continue
        used = max(used, int(parts[2].split()[0]) / 1024)
        total = max(total, int(parts[3].split()[0]) / 1024)
    return (used, total) if total else None


def status() -> None:
    for path in sorted(RUNS.glob("**/metrics.jsonl")):
        run = path.parent
        steps = [r for r in records(run) if r["kind"] == "step"]
        state = "FINITO" if (run / "checkpoints" / "final.pt").is_file() else "in corso"
        if (run / "FAILED").is_file():
            state = f"FALLITO: {(run / 'FAILED').read_text().strip()}"
        if not steps:
            print(f"{run}: nessun passo ancora; {state}")
            continue
        last = steps[-1]
        eta = last.get("eta_seconds") or 0
        print(
            f"{run}: passo {last['step']}, epoca {last['epoch'] + 1}, stage {last['stage']}, "
            f"{last['seconds']:.1f} s/passo, {last['memory_gib']:.1f} GiB, "
            f"fine stimata fra {eta / 86400:.1f} giorni; {state}"
        )


def real_plan() -> tuple[int, dict[str, int]]:
    """Passi per epoca ed epoche per stage del run vero."""
    from signworld.experiment.train.config import load_config  # noqa: PLC0415 (torch)

    config = load_config(*map(Path, REAL))
    assert config.data.index is not None
    splits = pq.read_table(config.data.index, columns=["split"]).column("split")
    train = pc.sum(pc.equal(splits, "train")).as_py()
    training = config.training
    stages = training.stages
    epochs = {
        "P": stages.pose_epochs,
        "F0": stages.heads_epochs,
        "F": training.epochs - stages.pose_epochs - stages.heads_epochs,
    }
    return train // training.batch_size, epochs


def debug() -> None:
    per_epoch, epochs = real_plan()
    print(f"run vero: {per_epoch} passi per epoca, epoche per stage {epochs}\n")
    for run in sorted(p for p in (RUNS / "debug").iterdir() if p.is_dir()):
        logs = sorted(Path("out").glob(f"sw-debug-{run.name}-*.log"))
        oom = any(OOM.search(log.read_text(errors="ignore")) for log in logs)
        by_stage: dict[str, list[dict]] = defaultdict(list)
        for record in records(run):
            if record["kind"] == "step":
                by_stage[record["stage"]].append(record)
        peak = gpu_peak(run)
        verdict = "OOM" if oom else "nessun dato"
        if peak and not oom:
            used, total = peak
            verdict = "OK" if used <= SAFE_SHARE * total else "RISCHIO OOM"
        if "F" not in by_stage and not oom:
            verdict += " (stage F non raggiunto: misura incompleta)"
        print(f"== {run.name}: {verdict}")
        if peak:
            print(f"   picco GPU (nvidia-smi): {peak[0]:.1f} di {peak[1]:.1f} GiB")
        days = 0.0
        for stage in ("P", "F0", "F"):
            rows = by_stage.get(stage)
            if not rows:
                print(f"   {stage}: nessun passo")
                continue
            seconds = statistics.median(r["seconds"] for r in rows)
            memory = max(r["memory_gib"] for r in rows)
            days += seconds * per_epoch * epochs[stage] / 86400
            # max_memory_allocated è cumulativo: comprende gli stage prima e il preflight P13.
            print(f"   {stage}: {seconds:.1f} s/passo, picco finora GPU 0 {memory:.1f} GiB")
        if all(s in by_stage for s in ("P", "F0", "F")):
            print(f"   run vero, solo passi: {days:.1f} giorni su un nodo")
        if (run / "FAILED").is_file():
            print(
                f"   FAILED: {(run / 'FAILED').read_text().strip()}; log: {[str(p) for p in logs]}"
            )


if __name__ == "__main__":
    {"status": status, "debug": debug}[sys.argv[1] if len(sys.argv) > 1 else "status"]()

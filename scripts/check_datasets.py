"""Controllo dei dataset prima del training: download, manifest, embedding, clip e pose, indice.

Uso (dalla root del repo):
    uv run --no-sync python scripts/check_datasets.py [--deep] [--workers 64] \
        [--source youtube_sl25 --source csl_daily --source phoenix14t] [--json out.json]

Senza --deep: solo file e conteggi (secondi). Con --deep: ffprobe di ogni video grezzo e di ogni
clip, e lettura di ogni file di posa (minuti; parallelo su --workers processi).
Le liste dei file con problemi finiscono in --problems-dir (una per tipo di problema).
"""

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

MIN_S, MAX_S = 1.0, 20.0  # test_data.{min,max}_duration_s di parameters/analysis/default.yaml
POSE_SHAPE = (64, 133, 3)


@dataclass(frozen=True)
class Dataset:
    name: str
    data: Path
    training: Path
    text: str
    min_s: float = MIN_S


DATASETS = {
    "youtube_sl25": Dataset(
        "youtube_sl25", Path("data/youtube-sl-25"), Path("data/training/youtube_sl25"),
        "2fdc70f119730c4c",
    ),
    "csl_daily": Dataset(
        "csl_daily", Path("data/csl_daily"), Path("data/training/csl_daily"),
        "2fdc70f119730c4c", 0.0,
    ),
    "phoenix14t": Dataset(
        "phoenix14t", Path("data/phoenix14t"), Path("data/training/phoenix14t"),
        "2fdc70f119730c4c", 0.0,
    ),
}  # fmt: skip


def probe(path: str) -> tuple[str, str | None, int, str]:
    """(path, codec, frames, error) letti decodificando l'intero flusso video."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=codec_name,nb_read_packets", "-of", "json", path,
    ]  # fmt: skip
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    except subprocess.TimeoutExpired:
        return path, None, 0, "timeout"
    if result.returncode != 0:
        return path, None, 0, result.stderr.strip()[:200] or f"rc={result.returncode}"
    streams = json.loads(result.stdout or "{}").get("streams", [])
    if not streams:
        return path, None, 0, "nessun flusso video"
    stream = streams[0]
    return path, stream.get("codec_name"), int(stream.get("nb_read_packets", 0)), ""


def check_pose(item: tuple[str, int]) -> tuple[str, str]:
    """(path, errore o '') di un file di posa rispetto ai frame della clip."""
    path, frames = item
    try:
        with np.load(path) as z:
            pose, idx, total = z["pose"], z["idx"], int(z["total"])
    except Exception as error:  # qualsiasi file illeggibile e' un problema
        return path, f"illeggibile: {error}"[:200]
    if pose.shape != POSE_SHAPE:
        return path, f"forma {pose.shape}"
    if not np.isfinite(pose.astype(np.float32)).all():
        return path, "valori non finiti"
    if total != frames:
        return path, f"total {total} != frames {frames}"
    if idx.min() < 0 or idx.max() >= total:
        return path, "idx fuori dalla clip"
    return path, ""


def ledger(data: Path) -> Counter[str]:
    latest: dict[str, tuple[str, str]] = {}
    for path in sorted((data / "ledgers").glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                if row["key"] not in latest or row["recorded_at"] >= latest[row["key"]][0]:
                    latest[row["key"]] = (row["recorded_at"], row["status"])
    return Counter(status for _, status in latest.values())


def materialized(training: Path, problems: dict[str, list[str]]) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in sorted(training.glob("index*.jsonl")):
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            problems["indice_senza_newline_finale"].append(str(path))
        for number, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                problems["indice_riga_rotta"].append(f"{path}:{number}")
                continue
            if record["clip_id"] in records:
                problems["indice_clip_duplicata"].append(record["clip_id"])
            records[record["clip_id"]] = record
    return records


def audit(ds: Dataset, deep: bool, workers: int) -> tuple[dict, dict[str, list[str]]]:  # noqa: PLR0912, PLR0915 (un controllo dopo l'altro)
    problems: defaultdict[str, list[str]] = defaultdict(list)
    out: dict = {"dataset": ds.name}

    # 1. download
    out["ledger"] = dict(ledger(ds.data))
    videos_dir = ds.data / "raw" / "videos"
    files = list(videos_dir.iterdir()) if videos_dir.is_dir() else []
    mp4 = [f for f in files if f.suffix == ".mp4" and ".partial" not in f.name]
    out["raw_files"] = dict(Counter(f.suffix for f in files))
    for f in files:
        if ".partial" in f.name or f.suffix in {".part", ".ytdl", ".temp"}:
            problems["raw_residui_parziali"].append(str(f))
    for f in mp4:
        if f.stat().st_size == 0:
            problems["raw_video_vuoto"].append(str(f))

    # 2. manifest
    manifest = pq.read_table(ds.data / "manifest" / "clips.parquet").to_pylist()
    out["manifest_clips"] = len(manifest)
    out["manifest_videos"] = len({r["video_id"] for r in manifest})
    out["manifest_available"] = sum(r["video_available"] for r in manifest)
    out["manifest_splits"] = dict(Counter(r["split"] for r in manifest))
    on_disk = {f.stem for f in mp4}
    for r in manifest:
        if r["video_available"] and r["video_id"] not in on_disk:
            problems["manifest_disponibile_ma_video_assente"].append(r["clip_id"])
    unused = on_disk - {r["video_id"] for r in manifest}
    out["raw_videos_not_in_manifest"] = len(unused)
    missing_captions = [r["clip_id"] for r in manifest if not (r["caption"] or "").strip()]
    problems["manifest_didascalia_vuota"].extend(missing_captions)

    # 3. embedding
    store = ds.data / "text" / ds.text
    if (store / "embeddings.npy").is_file():
        emb = np.load(store / "embeddings.npy", mmap_mode="r")
        captions = pq.read_table(store / "captions.parquet").num_rows
        clips = pq.read_table(store / "clips.parquet", columns=["clip_id"]).column(0).to_pylist()
        out["embeddings"] = {"rows": int(emb.shape[0]), "dim": int(emb.shape[1]),
                             "captions": captions, "clips": len(clips)}  # fmt: skip
        if emb.shape[0] != captions or set(clips) != {r["clip_id"] for r in manifest}:
            problems["embedding_incoerenti_col_manifest"].append(str(store))
        sample = np.asarray(emb[: min(len(emb), 20000)], dtype=np.float32)
        if not np.isfinite(sample).all():
            problems["embedding_non_finiti"].append(str(store))
    else:
        problems["embedding_mancanti"].append(str(store))

    # 4. clip e pose
    records = materialized(ds.training, problems)
    out["materialized"] = len(records)
    eligible = {
        r["clip_id"]: r for r in manifest
        if r["video_available"] and ds.min_s <= r["end_s"] - r["start_s"] <= MAX_S
    }  # fmt: skip
    out["eligible"] = len(eligible)
    out["out_of_duration"] = sum(
        1 for r in manifest if r["video_available"] and r["clip_id"] not in eligible
    )
    not_done = sorted(set(eligible) - set(records))
    out["eligible_not_materialized"] = len(not_done)
    problems["clip_idonee_non_materializzate"].extend(not_done)
    out["materialized_not_in_manifest"] = len(set(records) - {r["clip_id"] for r in manifest})
    for cid, rec in records.items():
        for key in ("video", "pose"):
            p = Path(rec[key])
            if not p.is_file():
                problems[f"{key}_mancante"].append(cid)
            elif p.stat().st_size == 0:
                problems[f"{key}_vuoto"].append(cid)
    for sub in ("clips", "poses"):
        d = ds.training / sub
        n = sum(1 for _ in d.iterdir()) if d.is_dir() else 0
        out[f"{sub}_files"] = n

    # 5. indice di training
    index_path = ds.training / "index.parquet"
    if index_path.is_file():
        index = pq.read_table(index_path, columns=["clip_id", "split"]).to_pylist()
        out["index_rows"] = len(index)
        out["index_splits"] = dict(Counter(r["split"] for r in index))
        newest = max((p.stat().st_mtime for p in ds.training.glob("index*.jsonl")), default=0)
        out["index_stale"] = index_path.stat().st_mtime < newest or (
            (store / "embeddings.npy").is_file()
            and index_path.stat().st_mtime < (store / "embeddings.npy").stat().st_mtime
        )
        out["materialized_not_in_index"] = len(set(records) - {r["clip_id"] for r in index})
    else:
        problems["indice_training_mancante"].append(str(index_path))

    # 6. verifica profonda
    if deep:
        with ProcessPoolExecutor(workers) as pool:
            codecs: Counter[str] = Counter()
            for path, codec, frames, error in pool.map(probe, map(str, mp4), chunksize=16):
                codecs[codec or "?"] += 1
                if error:
                    problems["raw_video_illeggibile"].append(f"{path}\t{error}")
                elif frames < 2:  # noqa: PLR2004 (servono due frame per il moto)
                    problems["raw_video_senza_frame"].append(path)
            out["raw_codecs"] = dict(codecs)
            clip_codecs: Counter[str] = Counter()
            by_path = {rec["video"]: rec for rec in records.values()}
            for path, codec, frames, error in pool.map(probe, list(by_path), chunksize=64):
                clip_codecs[codec or "?"] += 1
                if error:
                    problems["clip_illeggibile"].append(f"{path}\t{error}")
                elif frames != by_path[path]["frames"]:
                    problems["clip_frame_diversi_da_indice"].append(
                        f"{path}\t{frames} != {by_path[path]['frames']}"
                    )
            out["clip_codecs"] = dict(clip_codecs)
            items = [(rec["pose"], rec["frames"]) for rec in records.values()]
            for path, error in pool.map(check_pose, items, chunksize=256):
                if error:
                    problems["posa_non_valida"].append(f"{path}\t{error}")
    return out, {k: v for k, v in problems.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", choices=list(DATASETS))
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--problems-dir", type=Path, default=Path("reports/dataset_check"))
    args = parser.parse_args()
    report = {}
    for name in args.source or list(DATASETS):
        summary, problems = audit(DATASETS[name], args.deep, args.workers)
        summary["problems"] = {k: len(v) for k, v in problems.items()}
        report[name] = summary
        folder = args.problems_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.txt"):
            old.unlink()
        for kind, items in problems.items():
            (folder / f"{kind}.txt").write_text("\n".join(items) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

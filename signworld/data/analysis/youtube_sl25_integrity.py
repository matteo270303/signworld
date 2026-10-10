"""Integrity checks for a retrieved YouTube-SL-25: videos, subtitles, and their pairing.

Run after the downloads have finished::

    python -m signworld.data.analysis.youtube_sl25_integrity --root ~/dataset/youtube-sl-25 \\
        --out reports/youtube_sl25

Every check yields a list of offending IDs or paths; ``integrity.json`` holds them all, and
``--quarantine`` moves the files that cannot be used (unreadable video, empty subtitle) aside
instead of deleting them. Nothing is modified without that flag.
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from signworld.data.analysis import youtube_sl25 as sl

logger = logging.getLogger(__name__)

TEMP_SUFFIXES = (".part", ".ytdl", ".temp", ".tmp")
FORMAT_FRAGMENT = re.compile(r"\.f\d+\.(mp4|webm|m4a)$")
DURATION_TOLERANCE_S = 2.0
"""Probed vs. YouTube-reported duration; beyond this the file is truncated or the wrong video."""
CUE_OVERRUN_S = 5.0


def _scan(videos: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(videos.iterdir()):
        groups["temp" if path.name.endswith(TEMP_SUFFIXES) or FORMAT_FRAGMENT.search(path.name)
               else "video" if sl.VIDEO_NAME.match(path.name)
               else "subtitle" if sl.SUBTITLE_NAME.match(path.name)
               else "other"].append(path)  # fmt: skip
    return groups


def check(root: Path, out: Path, workers: int = 32) -> dict[str, object]:
    """All checks; returns a JSON-serialisable report."""
    release = sl.read_release(root / "youtube-sl-25-metadata.csv")
    in_release = {video_id for video_id, _ in release}
    groups = _scan(root / "videos")
    probe = sl.probe_directory(root / "videos", out / "probe.parquet", workers)
    probed = {row["video_id"]: row for row in probe}

    videos_by_id: dict[str, list[Path]] = defaultdict(list)
    for path in groups["video"]:
        videos_by_id[sl.VIDEO_NAME.match(path.name)["id"]].append(path)
    subs_by_id: dict[str, list[Path]] = defaultdict(list)
    for path in groups["subtitle"]:
        subs_by_id[sl.SUBTITLE_NAME.match(path.name)["id"]].append(path)

    report: dict[str, object] = {
        "counts": {
            "release_rows": len(release),
            "release_unique_ids": len(in_release),
            "video_files": len(groups["video"]),
            "subtitle_files": len(groups["subtitle"]),
            "temp_files": len(groups["temp"]),
            "other_files": len(groups["other"]),
        },
        "temp_files": [p.name for p in groups["temp"]],
        "other_files": [p.name for p in groups["other"]],
        "release_duplicate_ids": sl.duplicate_ids(release),
    }

    # ── videos ──
    report["video_multiple_containers"] = {
        i: [p.name for p in ps] for i, ps in videos_by_id.items() if len(ps) > 1
    }
    report["video_not_in_release"] = sorted(set(videos_by_id) - in_release)
    report["video_unreadable"] = {
        i: probed[i]["error"] for i in videos_by_id if i in probed and not probed[i]["ok"]
    }
    report["video_zero_size"] = [i for i, r in probed.items() if r["size_bytes"] == 0]
    report["video_near_duplicates"] = sl.near_duplicates(probe)
    report["video_anomalies"] = sl.anomalies(probe)

    # content hash duplicates, restricted to files sharing size (cheap pre-filter)
    by_size: dict[int, list[Path]] = defaultdict(list)
    for path in groups["video"]:
        by_size[path.stat().st_size].append(path)
    identical = []
    for paths in (ps for ps in by_size.values() if len(ps) > 1):
        digests: dict[str, list[str]] = defaultdict(list)
        for path in paths:
            h = hashlib.sha1()
            with path.open("rb") as stream:
                h.update(stream.read(1 << 22))
            digests[h.hexdigest()].append(path.name)
        identical += [names for names in digests.values() if len(names) > 1]
    report["video_identical_content"] = identical

    # ── subtitles ──
    empty, unparsable, overrun, bad_lang, duplicate_text = [], [], [], [], defaultdict(list)
    for video_id, paths in subs_by_id.items():
        duration = probed.get(video_id, {}).get("duration_s", float("nan"))
        for path in paths:
            raw = path.read_bytes()
            if not raw.strip():
                empty.append(path.name)
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                unparsable.append(f"{path.name}: not utf-8")
                text = raw.decode("utf-8", errors="replace")
            cues = [c for c in sl.parse_cues(text) if c[1] > c[0]]
            if not cues:
                unparsable.append(f"{path.name}: no cues")
                continue
            if duration == duration and max(c[1] for c in cues) > duration + CUE_OVERRUN_S:
                overrun.append(path.name)
            if "�" in text:
                unparsable.append(f"{path.name}: replacement characters")
            body = " ".join(c[2] for c in cues)
            duplicate_text[hashlib.sha1(body.encode()).hexdigest()].append(path.name)
    report["subtitle_empty"] = empty
    report["subtitle_unparsable"] = unparsable
    report["subtitle_cues_beyond_video_end"] = overrun
    report["subtitle_identical_text_across_videos"] = [
        n for n in duplicate_text.values()
        if len({sl.SUBTITLE_NAME.match(x)["id"] for x in n}) > 1 and len(" ".join(n)) > 0
    ]  # fmt: skip
    report["subtitle_without_video"] = sorted(set(subs_by_id) - set(videos_by_id))

    # ── pairing ──
    valid = {i for i in videos_by_id if i in probed and probed[i]["ok"]}
    usable_subs = {
        i for i, ps in subs_by_id.items()
        if any(p.name not in empty and not any(p.name in u for u in unparsable) for p in ps)
    }  # fmt: skip
    report["video_without_subtitle"] = sorted(valid - set(subs_by_id))
    report["video_without_subtitle_but_with_info"] = sorted(
        (valid - set(subs_by_id)) & {p.stem for p in (root / "info").glob("*.json")}
    )
    report["video_without_usable_subtitle"] = sorted(valid - usable_subs)
    report["paired"] = len(valid & usable_subs)

    # ── info json ──
    info_dir = root / "info"
    infos = {}
    for path in info_dir.glob("*.json"):
        if path.name.endswith(".info.json"):
            continue
        try:
            infos[path.stem] = json.loads(path.read_text())
        except json.JSONDecodeError:
            report.setdefault("info_corrupt", []).append(path.name)
    report["video_without_info"] = sorted(set(videos_by_id) - set(infos))
    mismatch = []
    for i, info in infos.items():
        if i in valid and info.get("duration"):
            if abs(info["duration"] - probed[i]["duration_s"]) > DURATION_TOLERANCE_S:
                mismatch.append([i, info["duration"], round(probed[i]["duration_s"], 1)])
    report["duration_mismatch_info_vs_file"] = mismatch
    declared_missing = []
    for i, info in infos.items():
        langs = set(info.get("subtitle_languages") or []) - {"live_chat"}
        have = {sl.SUBTITLE_NAME.match(p.name)["lang"] for p in subs_by_id.get(i, [])}
        if langs - have:
            declared_missing.append([i, sorted(langs - have)])
    report["subtitle_declared_but_missing"] = declared_missing
    return report


def quarantine(root: Path, report: dict[str, object]) -> list[str]:
    """Move unusable files to ``quarantine_integrity/`` and return their names."""
    target = root / "quarantine_integrity"
    target.mkdir(exist_ok=True)
    names = [f"{i}.mp4" for i in report["video_unreadable"]] + list(report["subtitle_empty"])
    moved = []
    for name in names:
        source = root / "videos" / name
        if source.is_file():
            shutil.move(source, target / name)
            moved.append(name)
    return moved


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--quarantine", action="store_true")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arguments.out.mkdir(parents=True, exist_ok=True)
    report = check(arguments.root, arguments.out, arguments.workers)
    if arguments.quarantine:
        report["quarantined"] = quarantine(arguments.root, report)
    (arguments.out / "integrity.json").write_text(json.dumps(report, indent=1, default=str))
    for key, value in report.items():
        size = len(value) if hasattr(value, "__len__") else value
        print(f"{key:45s} {size}")
    if isinstance(report["counts"], dict):
        print(report["counts"])


if __name__ == "__main__":
    main()

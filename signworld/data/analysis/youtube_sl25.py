"""YouTube-SL-25 on disk: what the release lists, what was retrieved, and what the files are.

Three layers, each reading the previous one:

* the release metadata (``video_id,language``), which says how the corpus is meant to be composed;
* the retrieval state of every ID (downloaded, corrupt, unavailable, blocked, not yet tried);
* a probe of every downloaded file (duration, resolution, frame rate, streams).

The statistics are plain numpy/scipy functions over arrays so they can be reused and tested
apart from the plotting. Only the release metadata is trusted for balance; everything measured
on the files describes the *retrieved* subset, which is biased by what YouTube still serves, so
``retrieval_bias`` compares the two.
"""

import csv
import json
import logging
import math
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

logger = logging.getLogger(__name__)

UNKNOWN: Final = "???"
VIDEO_NAME: Final = re.compile(r"^(?P<id>[A-Za-z0-9_-]{11})\.(?P<ext>mp4|mkv|webm|mov)$")
ISO_639_3: Final = re.compile(r"^[a-z]{3}$")
FAILURE_FILES: Final = {
    "unavailable": "failed_unavailable.txt",
    "bot": "failed_bot.txt",
    "other": "failed_other.txt",
}
STATES: Final = ("downloaded", "corrupt", "unavailable", "bot", "other", "untried")
"""Retrieval states, in the precedence used when an ID falls in several."""

CLAIMED_HOURS: Final = 3200.0
"""The release README: "more than 3200 hours of sign language videos"."""


# ── release metadata ──────────────────────────────────────────────────────────────────────


def read_release(path: Path) -> list[tuple[str, str]]:
    """Rows of the release CSV as ``(video_id, language)``; ``???`` is kept as ``UNKNOWN``."""
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [(row[0].strip(), row[1].strip()) for row in csv.reader(stream) if len(row) >= 2]
    if not rows:
        raise ValueError(f"{path}: no 'video_id,language' rows")
    return rows


def duplicate_ids(rows: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
    """IDs listed more than once, with every language they appear under."""
    seen: dict[str, list[str]] = defaultdict(list)
    for video_id, language in rows:
        seen[video_id].append(language)
    return {video_id: langs for video_id, langs in seen.items() if len(langs) > 1}


# ── retrieval state ───────────────────────────────────────────────────────────────────────


def read_failures(roots: Iterable[Path]) -> dict[str, set[str]]:
    """Failed IDs by cause, merged over the given directories.

    The downloader overwrites these lists at the end of every run, so the union over the
    current files and their backups is the only complete record. A cause can be wrong in an old
    run (an expired cookie turns "bot check" into "unavailable"), which is why ``bot`` and
    ``other`` outrank nothing: ``classify`` only trusts them for IDs with no file on disk.
    """
    merged: dict[str, set[str]] = {cause: set() for cause in FAILURE_FILES}
    for root in roots:
        for cause, name in FAILURE_FILES.items():
            path = root / name
            if path.is_file():
                merged[cause].update(path.read_text().split())
    return merged


def classify(
    video_ids: Iterable[str], valid: set[str], broken: set[str], failures: Mapping[str, set[str]]
) -> dict[str, str]:
    """State of every ID, by the precedence of ``STATES``."""
    state: dict[str, str] = {}
    for video_id in video_ids:
        if video_id in valid:
            state[video_id] = "downloaded"
        elif video_id in broken:
            state[video_id] = "corrupt"
        else:
            state[video_id] = next(
                (cause for cause in ("unavailable", "bot", "other") if video_id in failures[cause]),
                "untried",
            )
    return state


# ── probing the files ─────────────────────────────────────────────────────────────────────

PROBE_COLUMNS: Final = (
    "video_id",
    "ext",
    "size_bytes",
    "mtime_ns",
    "ok",
    "error",
    "duration_s",
    "width",
    "height",
    "fps",
    "video_codec",
    "audio_codec",
    "has_video",
    "has_audio",
    "bit_rate",
)


def _fraction(text: str | None) -> float:
    try:
        numerator, denominator = (text or "0/1").split("/")
        return float(numerator) / float(denominator) if float(denominator) else 0.0
    except ValueError:
        return 0.0


def _rotation(stream: Mapping[str, object]) -> int:
    for item in stream.get("side_data_list", ()):  # type: ignore[attr-defined]
        if "rotation" in item:
            return int(item["rotation"])
    tags = stream.get("tags") or {}
    return int(tags.get("rotate", 0)) if isinstance(tags, dict) else 0  # type: ignore[arg-type]


def probe_file(path: Path) -> dict[str, object]:
    """One row of ``PROBE_COLUMNS`` from ffprobe; failures are rows with ``ok=False``."""
    status = path.stat()
    match = VIDEO_NAME.match(path.name)
    assert match is not None
    row: dict[str, object] = {
        "video_id": match["id"],
        "ext": match["ext"],
        "size_bytes": status.st_size,
        "mtime_ns": status.st_mtime_ns,
        "ok": False,
        "error": "",
        "duration_s": float("nan"),
        "width": 0,
        "height": 0,
        "fps": float("nan"),
        "video_codec": "",
        "audio_codec": "",
        "has_video": False,
        "has_audio": False,
        "bit_rate": float("nan"),
    }
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
             str(path)],
            capture_output=True, text=True, timeout=120, check=False,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        row["error"] = "ffprobe timeout"
        return row
    if result.returncode != 0 or not result.stdout.strip():
        row["error"] = (result.stderr.strip().splitlines() or ["ffprobe failed"])[-1][:200]
        return row
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    row["has_video"], row["has_audio"] = video is not None, audio is not None
    row["audio_codec"] = (audio or {}).get("codec_name", "")
    row["duration_s"] = float(info.get("format", {}).get("duration") or "nan")
    row["bit_rate"] = float(info.get("format", {}).get("bit_rate") or "nan")
    if video is not None:
        width, height = int(video.get("width", 0)), int(video.get("height", 0))
        if abs(_rotation(video)) % 180 == 90:
            width, height = height, width
        row.update(
            width=width,
            height=height,
            video_codec=video.get("codec_name", ""),
            fps=_fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate")),
        )
    row["ok"] = bool(video is not None and audio is not None and row["duration_s"] > 0)
    if not row["ok"]:
        row["error"] = "no video stream" if video is None else (
            "no audio stream" if audio is None else "zero duration"
        )  # fmt: skip
    return row


def probe_directory(videos: Path, cache: Path, workers: int = 32) -> list[dict[str, object]]:
    """Probe every ``<id>.<ext>`` in ``videos``; unchanged files are read from ``cache``.

    A file is re-probed when its size or modification time changed, so a download that
    finishes after a first pass is picked up by the next one.
    """
    files = sorted(p for p in videos.iterdir() if VIDEO_NAME.match(p.name))
    known: dict[str, dict[str, object]] = {}
    if cache.is_file():
        known = {row["video_id"]: row for row in pq.read_table(cache).to_pylist()}
    fresh, todo = [], []
    for path in files:
        status = path.stat()
        cached = known.get(path.name.split(".")[0])
        if (
            cached is not None
            and cached["size_bytes"] == status.st_size
            and cached["mtime_ns"] == status.st_mtime_ns
        ):
            fresh.append(cached)
        else:
            todo.append(path)
    logger.info("probing %d files (%d cached)", len(todo), len(fresh))
    with ThreadPoolExecutor(workers) as pool:
        fresh.extend(pool.map(probe_file, todo))
    cache.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("video_id", pa.string()), ("ext", pa.string()), ("size_bytes", pa.int64()),
            ("mtime_ns", pa.int64()), ("ok", pa.bool_()), ("error", pa.string()),
            ("duration_s", pa.float64()), ("width", pa.int32()), ("height", pa.int32()),
            ("fps", pa.float64()), ("video_codec", pa.string()), ("audio_codec", pa.string()),
            ("has_video", pa.bool_()), ("has_audio", pa.bool_()), ("bit_rate", pa.float64()),
        ]
    )  # fmt: skip
    pq.write_table(pa.Table.from_pylist(fresh, schema=schema), cache)
    return fresh


# ── balance of a categorical distribution ─────────────────────────────────────────────────


def gini(counts: Sequence[float]) -> float:
    """Gini coefficient: 0 when every class has the same size, near 1 when one class has all."""
    values = np.sort(np.asarray(counts, dtype=float))
    n = values.size
    if n == 0 or values.sum() == 0:
        return float("nan")
    ranks = np.arange(1, n + 1)
    return float(2 * (ranks * values).sum() / (n * values.sum()) - (n + 1) / n)


def entropy_nats(counts: Sequence[float]) -> float:
    p = np.asarray(counts, dtype=float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


def normalised_entropy(counts: Sequence[float]) -> float:
    """Shannon entropy over its maximum: 1 is perfectly balanced, 0 is a single class."""
    return entropy_nats(counts) / math.log(len(counts)) if len(counts) > 1 else 0.0


def effective_classes(counts: Sequence[float]) -> float:
    """Inverse Simpson index: how many equally likely classes the distribution is worth."""
    p = np.asarray(counts, dtype=float)
    p = p / p.sum()
    return float(1.0 / (p**2).sum())


def lorenz(counts: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative share of items held by the smallest ``x`` share of classes."""
    values = np.sort(np.asarray(counts, dtype=float))
    cumulative = np.concatenate([[0.0], np.cumsum(values) / values.sum()])
    return np.linspace(0, 1, values.size + 1), cumulative


def zipf_fit(counts: Sequence[float]) -> tuple[float, float]:
    """Slope and R² of log count against log rank: the exponent of a Zipf-like long tail."""
    values = np.sort(np.asarray(counts, dtype=float))[::-1]
    values = values[values > 0]
    fit = stats.linregress(np.log(np.arange(1, values.size + 1)), np.log(values))
    return float(fit.slope), float(fit.rvalue**2)


def temperature_sampling(counts: Sequence[float], temperatures: Sequence[float]) -> list[dict]:
    """Language mix when sampling ``p_t ∝ n^t`` and how balanced each mix is.

    ``t=1`` is the corpus as it is, ``t=0`` is uniform over languages. The effective number of
    languages and the share of the largest language show what a multilingual run would see.
    """
    n = np.asarray(counts, dtype=float)
    rows = []
    for t in temperatures:
        p = n**t / (n**t).sum()
        rows.append(
            {
                "temperature": float(t),
                "effective_languages": effective_classes(p),
                "top_share": float(p.max()),
                "entropy_norm": normalised_entropy(p),
            }
        )
    return rows


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a proportion; defined for 0 successes and small totals."""
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return centre - half, centre + half


# ── retrieval bias ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetrievalBias:
    languages: list[str]
    downloaded: np.ndarray
    unavailable: np.ndarray
    chi2: float
    p_value: float
    cramers_v: float
    total_variation: float
    """Distance between the language mix of the release and of what was retrieved."""
    kl_nats: float


def retrieval_bias(
    languages: Sequence[str], state: Mapping[str, str], language_of: Mapping[str, str]
) -> RetrievalBias:
    """Does availability depend on the language?

    Among IDs whose fate is known (downloaded or unavailable), a chi-square test of
    independence between language and fate, with Cramér's V as its size. Languages with fewer
    than 10 resolved IDs are left out of the test, not of the other figures.
    """
    by_language: dict[str, list[int]] = {lang: [0, 0, 0] for lang in languages}
    for video_id, lang in language_of.items():
        counts = by_language[lang]
        counts[2] += 1
        if state[video_id] == "downloaded":
            counts[0] += 1
        elif state[video_id] == "unavailable":
            counts[1] += 1
    downloaded = np.array([by_language[lang][0] for lang in languages])
    unavailable = np.array([by_language[lang][1] for lang in languages])
    release = np.array([by_language[lang][2] for lang in languages])
    enough = (downloaded + unavailable) >= 10
    table = np.stack([downloaded[enough], unavailable[enough]])
    if table.shape[1] > 1 and table.sum() > 0:
        chi2, p_value, _, _ = stats.chi2_contingency(table)
        cramers = math.sqrt(chi2 / (table.sum() * (min(table.shape) - 1)))
    else:
        chi2 = p_value = cramers = float("nan")
    p_release = release / release.sum()
    p_got = downloaded / max(downloaded.sum(), 1)
    mask = p_got > 0
    return RetrievalBias(
        languages=list(languages),
        downloaded=downloaded,
        unavailable=unavailable,
        chi2=float(chi2),
        p_value=float(p_value),
        cramers_v=float(cramers),
        total_variation=float(0.5 * np.abs(p_release - p_got).sum()),
        kl_nats=float((p_got[mask] * np.log(p_got[mask] / p_release[mask])).sum()),
    )


# ── durations ─────────────────────────────────────────────────────────────────────────────


def duration_summary(seconds: np.ndarray) -> dict[str, float]:
    """Quantiles, log-normal fit and how concentrated the hours are in the longest videos."""
    seconds = seconds[np.isfinite(seconds) & (seconds > 0)]
    logs = np.log(seconds)
    mu, sigma = float(logs.mean()), float(logs.std(ddof=1))
    ordered = np.sort(seconds)[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    quantiles = np.quantile(seconds, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "videos": float(seconds.size),
        "hours": float(seconds.sum() / 3600),
        "mean_min": float(seconds.mean() / 60),
        **{f"p{int(q * 100):02d}_min": float(v / 60) for q, v in zip(
            [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], quantiles, strict=True)},
        "lognormal_mu": mu,
        "lognormal_sigma": sigma,
        "lognormal_ks": float(stats.kstest((logs - mu) / sigma, "norm").statistic),
        "skew_log": float(stats.skew(logs)),
        "hours_share_top1pct": float(cumulative[max(int(0.01 * seconds.size) - 1, 0)]),
        "hours_share_top10pct": float(cumulative[max(int(0.10 * seconds.size) - 1, 0)]),
    }  # fmt: skip


def duration_by_language(
    seconds: np.ndarray, languages: np.ndarray, minimum: int = 30
) -> dict[str, float]:
    """Do languages differ in video length? Kruskal-Wallis over those with enough videos.

    ``epsilon_squared`` is the share of rank variance explained by language (0 to 1).
    """
    groups = [seconds[languages == lang] for lang in np.unique(languages)]
    groups = [g for g in groups if g.size >= minimum]
    if len(groups) < 2:
        return {"groups": float(len(groups)), "h": float("nan"), "p": float("nan"),
                "epsilon_squared": float("nan")}  # fmt: skip
    h, p = stats.kruskal(*groups)
    n = sum(g.size for g in groups)
    return {"groups": float(len(groups)), "h": float(h), "p": float(p),
            "epsilon_squared": float(h * (n + 1) / (n**2 - 1))}  # fmt: skip


def estimated_release_hours(
    seconds: np.ndarray,
    languages: np.ndarray,
    release_counts: Mapping[str, int],
    bootstrap: int = 500,
    seed: int = 0,
) -> dict[str, tuple[float, float, float]]:
    """Total hours each language would have if the unretrieved videos were like the retrieved.

    Mean retrieved duration times the language's release count, with a bootstrap 95 % interval
    over the retrieved videos. An *estimate under the assumption that availability does not
    depend on length*: compare its sum with the 3,200 h the release claims.
    """
    rng = np.random.default_rng(seed)
    result: dict[str, tuple[float, float, float]] = {}
    for lang, count in release_counts.items():
        sample = seconds[languages == lang]
        if sample.size < 2:
            continue
        means = rng.choice(sample, size=(bootstrap, sample.size)).mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
        result[lang] = (
            float(sample.mean() * count / 3600),
            float(low * count / 3600),
            float(high * count / 3600),
        )
    return result


# ── technical quality and anomalies ───────────────────────────────────────────────────────

HEIGHT_BUCKETS: Final = ((0, 240, "<240"), (240, 360, "240-359"), (360, 480, "360-479"),
                         (480, 720, "480-719"), (720, 1080, "720-1079"),
                         (1080, 10**6, ">=1080"))  # fmt: skip


def height_bucket(height: int) -> str:
    return next(label for low, high, label in HEIGHT_BUCKETS if low <= height < high)


def aspect_class(width: int, height: int) -> str:
    if not width or not height:
        return "unknown"
    ratio = width / height
    if ratio < 0.95:
        return "portrait"
    if ratio < 1.05:
        return "square"
    if ratio < 1.5:
        return "4:3-like"
    return "16:9-like" if ratio < 2.0 else "ultrawide"


def bits_per_pixel(bit_rate: np.ndarray, width: np.ndarray, height: np.ndarray, fps: np.ndarray):
    """Bits per pixel per frame: a codec-independent proxy of how much detail a video holds."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return bit_rate / (width * height * fps)


def anomalies(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Files worth a look, each with the reason; thresholds are ours and deliberately loose."""
    found: list[dict[str, object]] = []
    for row in rows:
        reasons = []
        if not row["ok"]:
            reasons.append(str(row["error"]))
        else:
            duration, fps, height = row["duration_s"], row["fps"], row["height"]
            if duration < 2:  # type: ignore[operator]
                reasons.append("duration < 2 s")
            if duration > 4 * 3600:  # type: ignore[operator]
                reasons.append("duration > 4 h")
            if fps < 10 or fps > 62:  # type: ignore[operator]
                reasons.append(f"fps {fps:.1f}")
            if height < 240:  # type: ignore[operator]
                reasons.append(f"height {height}")
        if reasons:
            found.append({"video_id": row["video_id"], "reason": "; ".join(reasons)})
    return found


def near_duplicates(rows: Sequence[Mapping[str, object]]) -> list[list[str]]:
    """Groups of different IDs with identical size and duration: re-uploads of one video."""
    groups: dict[tuple[int, float], list[str]] = defaultdict(list)
    for row in rows:
        if row["ok"]:
            groups[(row["size_bytes"], round(row["duration_s"], 3))].append(row["video_id"])  # type: ignore[arg-type,index]
    return [ids for ids in groups.values() if len(ids) > 1]


# ── subtitles, when they have been downloaded ─────────────────────────────────────────────

CUE_TIME: Final = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})"
)
SUBTITLE_NAME: Final = re.compile(
<<<<<<< HEAD
    r"^(?P<id>[A-Za-z0-9_-]{11})\.(?P<lang>[A-Za-z0-9_-]+)\.(?:vtt|srt)$"
=======
    r"^(?P<id>[A-Za-z0-9_-]{11})\.(?P<lang>[A-Za-z0-9-]+)\.(?:vtt|srt)$"
>>>>>>> refs/remotes/origin/analysis/youtube-sl25
)


def parse_cues(text: str) -> list[tuple[float, float, str]]:
    """``(start, end, text)`` of every cue of a WebVTT or SRT file."""
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "")):
        lines = block.strip().splitlines()
        for position, line in enumerate(lines):
            found = CUE_TIME.search(line)
            if found:
                g = found.groups(default="0")
                start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
                end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000
                body = re.sub(r"<[^>]+>", "", " ".join(lines[position + 1 :])).strip()
                cues.append((start, end, body))
                break
    return cues


def subtitle_stats(videos: Path, durations: Mapping[str, float]) -> list[dict[str, object]]:
    """Per subtitle file: cues, share of the video they cover, words per second, long cues."""
    rows = []
    for path in sorted(videos.iterdir()):
        match = SUBTITLE_NAME.match(path.name)
        if not match:
            continue
        cues = [c for c in parse_cues(path.read_text(errors="replace")) if c[1] > c[0]]
        if not cues:
            continue
        covered = 0.0
        last_end = -1.0
        for start, end, _ in sorted(cues):
            covered += max(end - max(start, last_end), 0.0)
            last_end = max(last_end, end)
        words = sum(len(text.split()) for _, _, text in cues)
        duration = durations.get(match["id"], float("nan"))
        rows.append(
            {
                "video_id": match["id"],
                "track": match["lang"],
                "cues": len(cues),
                "covered_s": covered,
                "coverage": covered / duration if duration > 0 else float("nan"),
                "words": words,
                "words_per_s": words / covered if covered else float("nan"),
                "cues_over_20s": sum(end - start > 20 for start, end, _ in cues),
            }
        )
    return rows

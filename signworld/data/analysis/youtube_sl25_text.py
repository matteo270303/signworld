"""Captions and channels of a retrieved YouTube-SL-25: clips, text statistics and splits.

The release carries no split. This module measures what the retrieved subset allows
(channels from ``info/*.json``, clips from the subtitle tracks), and writes a deterministic
split proposal by channel, following §3.4: never by clip, never by video.

Clips follow ``corpus.builders.YouTubeSL25Manifest``: one clip per cue of the single track
written in the video's own language (``own_language_track``).

    python -m signworld.data.analysis.youtube_sl25_text --root ~/dataset/youtube-sl-25 \\
        --out reports/youtube_sl25
"""

import argparse
import csv
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from signworld.data.analysis import youtube_sl25 as sl
from signworld.data.corpus.languages import own_language_track, written_language
from signworld.data.corpus.vtt import parse_vtt

logger = logging.getLogger(__name__)

NON_SPEECH = re.compile(r"^[\[\(♪♫\s].*[\]\)♪♫\s]$")
"""Cues like ``[Music]`` or ``♪``: annotations, not translations."""
SHORT_S, LONG_S = 1.0, 20.0
VALIDATION_FRACTION, TEST_FRACTION = 0.10, 0.10
MIN_CHANNELS_FOR_HELD_OUT = 10
"""A language gets held-out channels only with enough channels for a meaningful split."""


def _bucket(key: str, salt: str) -> float:
    """Deterministic number in [0, 1) from a key: the split survives re-runs and new data."""
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not len(values):
        return {}
    a = np.asarray(values, dtype=float)
    q = np.percentile(a, [1, 5, 25, 50, 75, 95, 99])
    return {
        "n": int(a.size), "mean": float(a.mean()), "p01": q[0], "p05": q[1], "p25": q[2],
        "p50": q[3], "p75": q[4], "p95": q[5], "p99": q[6], "max": float(a.max()),
    }  # fmt: skip


def gather(root: Path, probe_cache: Path) -> dict[str, object]:
    release = dict(reversed(sl.read_release(root / "youtube-sl-25-metadata.csv")))
    probe = {r["video_id"]: r for r in pq.read_table(probe_cache).to_pylist() if r["ok"]}
    videos = root / "videos"
    tracks: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in videos.iterdir():
        match = sl.SUBTITLE_NAME.match(path.name)
        if match:
            tracks[match["id"]][match["lang"]] = path
    info: dict[str, dict] = {}
    for path in (root / "info").glob("*.json"):
        if not path.name.endswith(".info.json"):
            info[path.stem] = json.loads(path.read_text())

    rows, clips = [], []
    for video_id, row in probe.items():
        language = release.get(video_id, sl.UNKNOWN)
        own = own_language_track(None if language == sl.UNKNOWN else language, tracks[video_id])
        meta = info.get(video_id, {})
        record = {
            "video_id": video_id,
            "sign_language": language,
            "channel_id": meta.get("channel_id"),
            "upload_year": (meta.get("upload_date") or "")[:4] or None,
            "duration_s": row["duration_s"],
            "has_info": bool(meta),
            "tracks": sorted(tracks[video_id]),
            "own_track": own,
            "written_language": written_language(own) if own else None,
        }
        rows.append(record)
        if own is None:
            continue
        cues = parse_vtt(tracks[video_id][own].read_text(encoding="utf-8", errors="replace"))
        for index, cue in enumerate(cues):
            clips.append(
                (video_id, index, cue.start_s, cue.end_s, cue.text, language, record["channel_id"])
            )
    return {"videos": rows, "clips": clips}


def text_statistics(data: dict[str, object]) -> dict[str, object]:
    videos, clips = data["videos"], data["clips"]
    out: dict[str, object] = {}
    out["videos_valid"] = len(videos)
    out["videos_with_info"] = sum(v["has_info"] for v in videos)
    out["videos_with_any_track"] = sum(bool(v["tracks"]) for v in videos)
    out["videos_with_own_track"] = sum(v["own_track"] is not None for v in videos)
    out["videos_unknown_sign_language"] = sum(v["sign_language"] == sl.UNKNOWN for v in videos)
    out["videos_tracks_but_no_own"] = sum(
        bool(v["tracks"]) and v["own_track"] is None for v in videos
    )
    out["tracks_per_video"] = dict(sorted(Counter(len(v["tracks"]) for v in videos).items()))
    out["own_track_names"] = Counter(v["own_track"] for v in videos if v["own_track"]).most_common(25)
    out["track_languages"] = Counter(t for v in videos for t in v["tracks"]).most_common(25)

    start = np.array([c[2] for c in clips]); end = np.array([c[3] for c in clips])
    duration = end - start
    words = np.array([len(c[4].split()) for c in clips])
    chars = np.array([len(c[4]) for c in clips])
    out["clips"] = len(clips)
    out["clip_hours"] = float(duration.sum() / 3600)
    out["clip_duration_s"] = _quantiles(duration)
    out["words_per_clip"] = _quantiles(words)
    out["chars_per_clip"] = _quantiles(chars)
    out["words_total"] = int(words.sum())
    out["words_per_second"] = _quantiles(words / np.maximum(duration, 1e-3))
    out["clips_shorter_than_1s"] = int((duration < SHORT_S).sum())
    out["clips_longer_than_20s"] = int((duration > LONG_S).sum())
    out["clips_non_positive_duration"] = int((duration <= 0).sum())
    out["clips_non_speech"] = sum(bool(NON_SPEECH.match(c[4])) for c in clips)
    out["clips_one_word"] = int((words == 1).sum())
    per_video = Counter(c[0] for c in clips)
    out["clips_per_video"] = _quantiles(list(per_video.values()))

    # overlaps inside a video and repeated texts
    by_video: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for video_id, _, s, e, text, *_ in clips:
        by_video[video_id].append((s, e, text))
    overlapping = repeated = beyond_end = 0
    length = {v["video_id"]: v["duration_s"] for v in videos}
    for video_id, cues in by_video.items():
        cues.sort()
        overlapping += sum(cues[i][1] > cues[i + 1][0] + 0.05 for i in range(len(cues) - 1))
        counts = Counter(t for *_, t in cues)
        repeated += sum(n - 1 for n in counts.values() if n > 1)
        beyond_end += sum(c[1] > length[video_id] + 5 for c in cues)
    out["clips_overlapping_next"] = overlapping
    out["clips_repeating_text_in_video"] = repeated
    out["clips_beyond_video_end"] = beyond_end
    covered_share = [
        sum(e - s for s, e, _ in c) / length[v] for v, c in by_video.items() if length[v] > 0
    ]
    out["cue_time_over_video_duration"] = _quantiles(covered_share)

    # per sign language and per written language
    per_language: dict[str, dict[str, float]] = defaultdict(
        lambda: {"videos": 0, "clips": 0, "hours": 0.0, "words": 0}
    )
    video_language = {v["video_id"]: v["sign_language"] for v in videos}
    for v in videos:
        per_language[v["sign_language"]]["videos"] += v["own_track"] is not None
    for (video_id, _, s, e, text, *_) in clips:
        row = per_language[video_language[video_id]]
        row["clips"] += 1; row["hours"] += (e - s) / 3600; row["words"] += len(text.split())  # noqa: E702
    out["per_sign_language"] = dict(sorted(per_language.items(), key=lambda kv: -kv[1]["clips"]))

    # identical captions across videos: re-uploads or templated texts
    signature: dict[str, set[str]] = defaultdict(set)
    for video_id, cues in by_video.items():
        signature[hashlib.sha1("\n".join(t for *_, t in sorted(cues)).encode()).hexdigest()].add(
            video_id
        )
    groups = [sorted(v) for v in signature.values() if len(v) > 1]
    out["identical_caption_groups"] = len(groups)
    out["identical_caption_videos"] = sum(len(g) for g in groups)
    out["identical_caption_examples"] = groups[:10]
    out["_identical_groups"] = groups

    pair = Counter((c[4], round(c[2], 1), round(c[3], 1)) for c in clips)
    out["clips_same_text_and_timing_elsewhere"] = sum(n for n in pair.values() if n > 1)
    return out


def channel_statistics(data: dict[str, object]) -> dict[str, object]:
    videos = [v for v in data["videos"] if v["channel_id"]]
    out: dict[str, object] = {"videos_with_channel": len(videos)}
    per_channel = Counter(v["channel_id"] for v in videos)
    counts = sorted(per_channel.values(), reverse=True)
    out["channels"] = len(per_channel)
    out["videos_per_channel"] = _quantiles(counts)
    out["gini_videos_per_channel"] = sl.gini(counts)
    out["top10_channels_share"] = sum(counts[:10]) / len(videos)
    out["top1pct_channels_share"] = sum(counts[: max(1, len(counts) // 100)]) / len(videos)
    out["channels_with_one_video"] = sum(n == 1 for n in counts)
    languages_of: dict[str, set[str]] = defaultdict(set)
    for v in videos:
        languages_of[v["channel_id"]].add(v["sign_language"])
    out["channels_spanning_languages"] = sum(len(s) > 1 for s in languages_of.values())
    channels_by_language: dict[str, set[str]] = defaultdict(set)
    for v in videos:
        channels_by_language[v["sign_language"]].add(v["channel_id"])
    out["channels_per_language"] = dict(
        sorted(((k, len(s)) for k, s in channels_by_language.items()), key=lambda kv: -kv[1])
    )
    out["upload_years"] = dict(sorted(Counter(v["upload_year"] for v in videos).items(),
                                      key=lambda kv: str(kv[0])))  # fmt: skip
    out["top_channels"] = [(c, n) for c, n in per_channel.most_common(10)]
    return out


def propose_splits(
    data: dict[str, object], identical_groups: list[list[str]], near_duplicates: list[list[str]]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Train / validation / test by channel, stratified by sign language.

    * channels are hashed within each language: no channel on two sides, deterministic;
    * channels linked by a duplicate (same file, or identical captions) are merged, so a
      re-upload never lands on the other side of its source;
    * a language with fewer than ``MIN_CHANNELS_FOR_HELD_OUT`` channels is train only;
    * videos without a channel (no metadata) are kept out of every split (``unassigned``).
    """
    videos = {v["video_id"]: v for v in data["videos"]}
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for v in videos.values():
        if v["channel_id"]:
            find(v["channel_id"])
    for group in [*identical_groups, *near_duplicates]:
        channels = {videos[i]["channel_id"] for i in group if i in videos and videos[i]["channel_id"]}
        for other in list(channels)[1:]:
            parent[find(other)] = find(next(iter(channels)))

    components: dict[str, set[str]] = defaultdict(set)
    language_of_component: dict[str, Counter] = defaultdict(Counter)
    for v in videos.values():
        if v["channel_id"]:
            root = find(v["channel_id"])
            components[root].add(v["channel_id"])
            language_of_component[root][v["sign_language"]] += 1
    n_components_by_language: Counter = Counter()
    main_language = {}
    for root, counter in language_of_component.items():
        main_language[root] = counter.most_common(1)[0][0]
        n_components_by_language[main_language[root]] += 1

    split_of_component = {}
    for root in components:
        language = main_language[root]
        if n_components_by_language[language] < MIN_CHANNELS_FOR_HELD_OUT:
            split_of_component[root] = "train"
            continue
        u = _bucket(root, language)
        split_of_component[root] = (
            "test" if u < TEST_FRACTION
            else "validation" if u < TEST_FRACTION + VALIDATION_FRACTION else "train"
        )  # fmt: skip

    rows = []
    for v in videos.values():
        split = split_of_component[find(v["channel_id"])] if v["channel_id"] else "unassigned"
        rows.append(
            {"video_id": v["video_id"], "channel_id": v["channel_id"] or "",
             "sign_language": v["sign_language"], "split": split}
        )  # fmt: skip

    clips_of = Counter(c[0] for c in data["clips"])
    hours_of: dict[str, float] = defaultdict(float)
    for c in data["clips"]:
        hours_of[c[0]] += (c[3] - c[2]) / 3600
    summary: dict[str, object] = {}
    for name in ("train", "validation", "test", "unassigned"):
        ids = [r["video_id"] for r in rows if r["split"] == name]
        summary[name] = {
            "videos": len(ids),
            "channels": len({videos[i]["channel_id"] for i in ids if videos[i]["channel_id"]}),
            "clips": sum(clips_of[i] for i in ids),
            "clip_hours": sum(hours_of[i] for i in ids),
            "video_hours": sum(videos[i]["duration_s"] for i in ids) / 3600,
            "languages": len({videos[i]["sign_language"] for i in ids}),
        }
    summary["languages_with_held_out"] = sorted(
        lang for lang, n in n_components_by_language.items() if n >= MIN_CHANNELS_FOR_HELD_OUT
    )
    summary["languages_train_only"] = sorted(
        lang for lang, n in n_components_by_language.items() if n < MIN_CHANNELS_FOR_HELD_OUT
    )
    summary["merged_channel_groups"] = sum(len(c) > 1 for c in components.values())

    # naive alternatives, to show what the channel split buys: how many validation videos
    # share a channel with training videos under a random split by video
    rng = np.random.default_rng(0)
    with_channel = [v for v in videos.values() if v["channel_id"]]
    pick = rng.random(len(with_channel)) < VALIDATION_FRACTION
    train_channels = {v["channel_id"] for v, p in zip(with_channel, pick) if not p}
    shared = sum(v["channel_id"] in train_channels for v, p in zip(with_channel, pick) if p)
    summary["random_video_split_validation_videos_sharing_train_channel"] = {
        "validation_videos": int(pick.sum()), "sharing_channel_with_train": shared,
    }
    return rows, summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = gather(arguments.root, arguments.out / "probe.parquet")
    text = text_statistics(data)
    identical = text.pop("_identical_groups")
    probe = pq.read_table(arguments.out / "probe.parquet").to_pylist()
    near = sl.near_duplicates(probe)
    channels = channel_statistics(data)
    rows, summary = propose_splits(data, identical, near)
    arguments.out.mkdir(parents=True, exist_ok=True)
    with (arguments.out / "splits_proposal.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"text": text, "channels": channels, "splits": summary}
    (arguments.out / "text_and_splits.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()

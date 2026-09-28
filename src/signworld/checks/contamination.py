"""Collaudo "contaminazione e duplicati" (§4.13.1) and assertion P2 (§4.13.2).

§3.9 removes from pre-training every segment of a corpus that, in the same video, overlaps a
validation or test clip of a benchmark (margin of 2 s), plus the captions identical or nearly
identical to validation or test captions of the same video. Removal is by interval, not by
video, because the benchmark's own training split already contains the rest of those videos.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Final

import pyarrow as pa

from ..corpus.text import NEAR_DUPLICATE_RATIO, are_near_duplicates, normalize_caption

EVALUATION_SPLITS: Final = frozenset({"valid", "dev", "test"})


@dataclass(frozen=True, slots=True)
class _Segment:
    clip_id: str
    start_s: float
    end_s: float
    caption: str


@dataclass(frozen=True, slots=True)
class ContaminationReport:
    corpus_clips: int
    evaluation_clips: int
    shared_videos: int
    """Videos present in the corpus that hold at least one evaluation clip."""
    overlapping: int
    near_duplicate_captions: int
    excluded: frozenset[str]
    """Corpus clip IDs to remove from pre-training: the union of both criteria."""


def _segments_by_video(table: pa.Table) -> dict[str, list[_Segment]]:
    grouped: dict[str, list[_Segment]] = defaultdict(list)
    columns = table.select(["video_id", "clip_id", "start_s", "end_s", "caption"]).to_pydict()
    for video_id, clip_id, start, end, caption in zip(*columns.values(), strict=True):
        grouped[video_id].append(_Segment(clip_id, start, end, caption))
    return grouped


def evaluation_clips(benchmark: pa.Table, splits: frozenset[str] = EVALUATION_SPLITS) -> pa.Table:
    mask = [split in splits for split in benchmark.column("split").to_pylist()]
    return benchmark.filter(pa.array(mask, type=pa.bool_()))


def contamination_report(
    corpus: pa.Table,
    benchmark: pa.Table,
    *,
    margin_s: float = 2.0,
    near_duplicate_ratio: float = NEAR_DUPLICATE_RATIO,
) -> ContaminationReport:
    """Corpus clips to exclude because they overlap or repeat benchmark evaluation clips."""
    evaluation = _segments_by_video(evaluation_clips(benchmark))
    corpus_segments = _segments_by_video(corpus)
    shared = corpus_segments.keys() & evaluation.keys()

    overlapping: set[str] = set()
    duplicated: set[str] = set()
    for video_id in shared:
        held_out = evaluation[video_id]
        for segment in corpus_segments[video_id]:
            if any(
                segment.start_s < other.end_s + margin_s
                and segment.end_s > other.start_s - margin_s
                for other in held_out
            ):
                overlapping.add(segment.clip_id)
            if any(
                are_near_duplicates(segment.caption, other.caption, near_duplicate_ratio)
                for other in held_out
            ):
                duplicated.add(segment.clip_id)

    return ContaminationReport(
        corpus_clips=corpus.num_rows,
        evaluation_clips=sum(len(clips) for clips in evaluation.values()),
        shared_videos=len(shared),
        overlapping=len(overlapping),
        near_duplicate_captions=len(duplicated),
        excluded=frozenset(overlapping | duplicated),
    )


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    caption: str
    duration_s: float
    splits: list[str]
    clip_ids: list[str]


def duplicates_across_splits(
    manifest: pa.Table, duration_tolerance_s: float = 0.1
) -> list[DuplicateGroup]:
    """Clips with the same normalised caption and duration that sit in different splits.

    Such pairs are typically one sentence uploaded twice to YouTube; left in place they put a
    training clip into validation or test.
    """
    columns = manifest.select(["clip_id", "start_s", "end_s", "caption", "split"]).to_pydict()
    groups: dict[tuple[str, int], list[tuple[str, str, float]]] = defaultdict(list)
    for clip_id, start, end, caption, split in zip(*columns.values(), strict=True):
        if split is None:
            continue
        key = (normalize_caption(caption), round((end - start) / duration_tolerance_s))
        groups[key].append((clip_id, split, end - start))
    return [
        DuplicateGroup(
            caption=caption,
            duration_s=members[0][2],
            splits=sorted({split for _, split, _ in members}),
            clip_ids=sorted(clip_id for clip_id, _, _ in members),
        )
        for (caption, _), members in sorted(groups.items())
        if caption and len({split for _, split, _ in members}) > 1
    ]

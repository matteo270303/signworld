"""The clip manifest: one row per captioned sentence clip of one dataset.

Datasets keep separate manifests; nothing here merges corpora. The Parquet schema is fixed so
every check and loader reads the same columns whatever the source.
"""

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA: Final = pa.schema(
    [
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("video_id", pa.string(), nullable=False),
        pa.field("start_s", pa.float64(), nullable=False),
        pa.field("end_s", pa.float64(), nullable=False),
        pa.field("caption", pa.string(), nullable=False),
        pa.field("caption_language", pa.string()),
        pa.field("sign_language", pa.string()),
        pa.field("channel_id", pa.string()),
        pa.field("split", pa.string()),
        pa.field("video_available", pa.bool_(), nullable=False),
    ]
)


@dataclass(frozen=True, slots=True)
class Clip:
    clip_id: str
    video_id: str
    start_s: float
    end_s: float
    caption: str
    caption_language: str | None
    sign_language: str | None
    channel_id: str | None
    split: str | None
    """Official split when the release defines one; ``None`` until our splits exist."""
    video_available: bool

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def write_manifest(clips: Iterable[Clip], path: Path) -> int:
    """Write the clips atomically and return how many were written."""
    table = pa.Table.from_pylist([asdict(clip) for clip in clips], schema=SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)
    return int(table.num_rows)


def read_manifest(path: Path) -> pa.Table:
    table = pq.read_table(path)
    if not table.schema.equals(SCHEMA):
        raise ValueError(f"{path}: schema differs from the manifest schema; rebuild the manifest")
    return table


def clips_of(table: pa.Table) -> list[Clip]:
    return [Clip(**row) for row in table.to_pylist()]

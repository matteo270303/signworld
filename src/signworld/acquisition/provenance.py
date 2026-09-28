"""Origins and checksums of the files a dataset release is built from."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

_CHUNK_BYTES: Final = 8 * 1024 * 1024


class ChecksumMismatchError(RuntimeError):
    """A downloaded file differs from the version the configuration pins."""


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    actual = sha256sum(path)
    if actual != expected.lower():
        raise ChecksumMismatchError(f"{path}: SHA-256 {actual} differs from the pinned {expected}")


@dataclass(frozen=True, slots=True)
class FileRecord:
    path: str
    origin: str
    sha256: str
    size_bytes: int
    retrieved_at: str


class ProvenanceLog:
    """``PROVENANCE.json`` of a metadata directory: origin and SHA-256 of every file.

    Anyone re-running the pipeline can compare checksums and know whether they built on
    the same release.
    """

    FILENAME: Final = "PROVENANCE.json"

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._path = directory / self.FILENAME

    def register(self, file: Path, origin: str) -> FileRecord:
        record = FileRecord(
            path=self._describe(file),
            origin=origin,
            sha256=sha256sum(file),
            size_bytes=file.stat().st_size,
            retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        records = {existing.path: existing for existing in self.records()}
        records[record.path] = record
        self._write(sorted(records.values(), key=lambda item: item.path))
        return record

    def records(self) -> list[FileRecord]:
        if not self._path.exists():
            return []
        return [FileRecord(**raw) for raw in json.loads(self._path.read_text(encoding="utf-8"))]

    def _describe(self, file: Path) -> str:
        """Path relative to the metadata directory when inside it, absolute otherwise."""
        resolved, directory = file.resolve(), self._directory.resolve()
        if resolved.is_relative_to(directory):
            return resolved.relative_to(directory).as_posix()
        return resolved.as_posix()

    def _write(self, records: list[FileRecord]) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps([asdict(record) for record in records], indent=2) + "\n"
        temporary = self._path.with_name(f"{self._path.name}.tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(self._path)

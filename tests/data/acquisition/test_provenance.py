import hashlib
from pathlib import Path

import pytest

from signworld.data.acquisition.provenance import (
    ChecksumMismatchError,
    ProvenanceLog,
    verify_sha256,
)


def test_registered_files_keep_origin_and_checksum(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    inside = metadata / "labels.json"
    inside.parent.mkdir()
    inside.write_bytes(b"[]")
    outside = tmp_path / "elsewhere.tsv"
    outside.write_bytes(b"vid\tyid\n")

    log = ProvenanceLog(metadata)
    log.register(inside, origin="https://example.org/labels.json")
    log.register(outside, origin="https://example.org/elsewhere.tsv")

    records = {record.path: record for record in ProvenanceLog(metadata).records()}
    assert records["labels.json"].sha256 == hashlib.sha256(b"[]").hexdigest()
    assert records[outside.resolve().as_posix()].origin == "https://example.org/elsewhere.tsv"


def test_pinned_checksum_mismatch_is_reported(tmp_path: Path) -> None:
    file = tmp_path / "metadata.csv"
    file.write_bytes(b"changed")

    with pytest.raises(ChecksumMismatchError):
        verify_sha256(file, hashlib.sha256(b"original").hexdigest())

from pathlib import Path

import pytest

from signworld.acquisition.config import HttpConfig
from signworld.acquisition.transfer.http import DownloadError, HttpDownloader

from .conftest import FileServer

PAYLOAD = bytes(range(256)) * 64


@pytest.fixture
def downloader() -> HttpDownloader:
    return HttpDownloader(HttpConfig(chunk_bytes=1024, retries=3, timeout_s=5.0, backoff_s=0.0))


def test_complete_file_is_published_without_partial_leftovers(
    file_server: FileServer, downloader: HttpDownloader, tmp_path: Path
) -> None:
    file_server.files["/data.bin"] = PAYLOAD

    target = downloader.fetch(
        file_server.url("/data.bin"), tmp_path / "data.bin", expected_bytes=len(PAYLOAD)
    )

    assert target.read_bytes() == PAYLOAD
    assert not (tmp_path / "data.bin.part").exists()


def test_partial_file_is_resumed_with_a_range_request(
    file_server: FileServer, downloader: HttpDownloader, tmp_path: Path
) -> None:
    file_server.files["/data.bin"] = PAYLOAD
    (tmp_path / "data.bin.part").write_bytes(PAYLOAD[:1000])

    downloader.fetch(file_server.url("/data.bin"), tmp_path / "data.bin")

    assert file_server.requests == [("/data.bin", "bytes=1000-")]
    assert (tmp_path / "data.bin").read_bytes() == PAYLOAD


def test_interrupted_transfer_resumes_on_retry(
    file_server: FileServer, downloader: HttpDownloader, tmp_path: Path
) -> None:
    file_server.files["/data.bin"] = PAYLOAD
    file_server.truncate_once.add("/data.bin")

    downloader.fetch(file_server.url("/data.bin"), tmp_path / "data.bin")

    assert len(file_server.requests) == 2
    assert file_server.requests[1][1] is not None
    assert (tmp_path / "data.bin").read_bytes() == PAYLOAD


def test_missing_file_fails_without_retrying(
    file_server: FileServer, downloader: HttpDownloader, tmp_path: Path
) -> None:
    with pytest.raises(DownloadError, match="404"):
        downloader.fetch(file_server.url("/missing.bin"), tmp_path / "missing.bin")

    assert len(file_server.requests) == 1


def test_size_mismatch_leaves_nothing_behind(
    file_server: FileServer, downloader: HttpDownloader, tmp_path: Path
) -> None:
    file_server.files["/data.bin"] = PAYLOAD

    with pytest.raises(DownloadError, match="expected"):
        downloader.fetch(
            file_server.url("/data.bin"), tmp_path / "data.bin", expected_bytes=len(PAYLOAD) + 1
        )

    assert list(tmp_path.iterdir()) == []


def test_complete_existing_file_is_not_requested_again(
    file_server: FileServer, downloader: HttpDownloader, tmp_path: Path
) -> None:
    target = tmp_path / "data.bin"
    target.write_bytes(PAYLOAD)

    downloader.fetch(file_server.url("/data.bin"), target, expected_bytes=len(PAYLOAD))

    assert file_server.requests == []
